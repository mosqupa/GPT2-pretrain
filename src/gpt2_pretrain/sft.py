from __future__ import annotations

import contextlib
from dataclasses import asdict
import math
import time
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from gpt2_pretrain.config import ProjectConfig, ensure_dirs
from gpt2_pretrain.data import SFTCollator, SFTJsonlDataset, format_sft_prompt
from gpt2_pretrain.tokenizer_utils import load_tokenizer
from gpt2_pretrain.trainer import (
    _cleanup_distributed,
    _create_session_run_dir,
    _find_latest_checkpoint,
    _is_cuda_device,
    _make_amp_context,
    _reduce_mean,
    _reduce_sum,
    _resolve_resume_path,
    _setup_distributed,
    _unwrap_model,
    build_model,
)
from gpt2_pretrain.utils import (
    append_csv,
    count_parameters,
    detect_dtype,
    save_json,
    set_seed,
)


def make_sft_dataloader(
    jsonl_path: str,
    tokenizer_path: str,
    seq_length: int,
    batch_size: int,
    num_workers: int,
    system_prompt: str,
    prompt_loss: bool,
    max_samples: int,
    distributed,
    shuffle: bool,
) -> tuple[DataLoader, DistributedSampler | None]:
    dataset = SFTJsonlDataset(jsonl_path, system_prompt=system_prompt, max_samples=max_samples)
    sampler = None
    if distributed.is_distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=distributed.world_size,
            rank=distributed.rank,
            shuffle=shuffle,
            drop_last=False,
        )
    collator = SFTCollator(tokenizer_path, seq_length=seq_length, prompt_loss=prompt_loss)
    return (
        DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle and sampler is None,
            sampler=sampler,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=_is_cuda_device(distributed.device),
            collate_fn=collator,
        ),
        sampler,
    )


@torch.no_grad()
def estimate_sft_loss(model, data_loader: DataLoader, device: str, amp_context, eval_iters: int, distributed) -> float:
    model.eval()
    losses = []
    data_iter = iter(data_loader)
    with torch.inference_mode():
        for _ in range(eval_iters):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                x, y = next(data_iter)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with amp_context():
                _, loss = model(x, y)
            losses.append(loss.item())
    model.train()
    loss_value = sum(losses) / len(losses)
    return _reduce_mean(loss_value, device, distributed)


def _load_sft_checkpoint(
    checkpoint_path: Path,
    raw_model,
    optimizer: torch.optim.Optimizer,
    device: str,
    run_name: str,
    distributed,
) -> tuple[int, float, int, bool]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    raw_model.load_state_dict(checkpoint["model_state"])

    checkpoint_config = checkpoint.get("config") or {}
    checkpoint_run_name = checkpoint_config.get("run_name")
    is_same_run = checkpoint_run_name == run_name
    if is_same_run and checkpoint.get("optimizer_state") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        step = int(checkpoint.get("step", 0))
        best_valid_loss = float(checkpoint.get("best_valid_loss", float("inf")))
        tokens_seen = int(checkpoint.get("tokens_seen", 0))
        if distributed.is_master:
            print(f"[resume] resumed SFT session from {checkpoint_path} at step={step}")
        return step, best_valid_loss, tokens_seen, True

    if distributed.is_master:
        print(f"[init] loaded pretrained weights from {checkpoint_path}, optimizer/step reset for SFT.")
    return 0, float("inf"), 0, False


def train_sft(config: ProjectConfig) -> dict | None:
    if config.sft is None:
        raise ValueError("SFT config is missing. Please add the `sft` section in YAML.")
    if not config.sft.train_file or not config.sft.valid_file:
        raise ValueError("SFT train_file/valid_file is required.")

    distributed = _setup_distributed(config.train.device)
    run_dir = _create_session_run_dir(config.run_name, distributed)
    config.paths.output_dir = str(run_dir)
    config.paths.checkpoint_dir = str(run_dir / "checkpoints")
    config.paths.sample_dir = str(run_dir / "samples")
    ensure_dirs(config)
    set_seed(config.seed)

    device = distributed.device
    device_type = "cuda" if _is_cuda_device(device) else "cpu"
    amp_dtype = detect_dtype(config.train.dtype, device_type)
    tokenizer = load_tokenizer(config.data.tokenizer_path)
    tokenizer_vocab_size = tokenizer.get_vocab_size()
    if tokenizer_vocab_size != config.model.vocab_size:
        if distributed.is_master:
            print(
                f"[warn] model.vocab_size({config.model.vocab_size}) != tokenizer_vocab_size({tokenizer_vocab_size}), "
                "auto-adjusting model vocab size."
            )
        config.model.vocab_size = tokenizer_vocab_size

    try:
        model = build_model(config).to(device)
        raw_model = model
        if distributed.is_distributed:
            ddp_kwargs = {"device_ids": [distributed.local_rank]} if device_type == "cuda" else {}
            model = DistributedDataParallel(model, **ddp_kwargs)

        train_loader, train_sampler = make_sft_dataloader(
            jsonl_path=config.sft.train_file,
            tokenizer_path=config.data.tokenizer_path,
            seq_length=config.data.seq_length,
            batch_size=config.train.batch_size,
            num_workers=config.train.num_workers,
            system_prompt=config.sft.system_prompt,
            prompt_loss=config.sft.prompt_loss,
            max_samples=config.sft.max_samples,
            distributed=distributed,
            shuffle=True,
        )
        valid_loader, _ = make_sft_dataloader(
            jsonl_path=config.sft.valid_file,
            tokenizer_path=config.data.tokenizer_path,
            seq_length=config.data.seq_length,
            batch_size=config.train.batch_size,
            num_workers=config.train.num_workers,
            system_prompt=config.sft.system_prompt,
            prompt_loss=config.sft.prompt_loss,
            max_samples=max(256, config.train.eval_iters * config.train.batch_size),
            distributed=distributed,
            shuffle=False,
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.train.learning_rate,
            betas=(config.train.beta1, config.train.beta2),
            weight_decay=config.train.weight_decay,
        )
        scaler = torch.amp.GradScaler(device_type, enabled=(device_type == "cuda" and amp_dtype == torch.float16))
        amp_context = lambda: _make_amp_context(device, amp_dtype)

        train_csv = Path(config.paths.output_dir) / "train_loss.csv"
        valid_csv = Path(config.paths.output_dir) / "valid_loss.csv"
        sample_txt = Path(config.paths.sample_dir) / "progress_samples.txt"
        metrics_json = Path(config.paths.output_dir) / "metrics.json"
        ckpt_last = Path(config.paths.checkpoint_dir) / "last.pt"
        ckpt_best = Path(config.paths.checkpoint_dir) / "best.pt"

        bos_id = tokenizer.token_to_id("[BOS]")
        sample_prompt = format_sft_prompt("请简要介绍人工智能。", "", config.sft.system_prompt)
        prompt_ids = [bos_id] + tokenizer.encode(sample_prompt, add_special_tokens=False).ids
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        best_val = float("inf")
        step = 0
        tokens_seen = 0
        start_step = 0
        round_steps = config.train.num_steps
        if round_steps <= 0:
            raise ValueError(f"train.num_steps must be > 0, got {round_steps}")

        resume_path = _resolve_resume_path(config)
        if resume_path is None:
            resume_path = _find_latest_checkpoint(config.run_name)
            if distributed.is_master and resume_path is not None:
                print(f"[resume] auto-resume from latest checkpoint: {resume_path}")
        if resume_path is not None:
            if not resume_path.exists():
                raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
            step, best_val, tokens_seen, resumed = _load_sft_checkpoint(
                resume_path, raw_model, optimizer, device, config.run_name, distributed
            )
            start_step = step if resumed else 0
        elif distributed.is_master:
            print("[resume] no checkpoint configured, SFT starts from scratch.")

        if distributed.is_master:
            print(f"[output] session run directory: {run_dir}")
        target_step = start_step + round_steps

        train_epoch = 0
        if train_sampler is not None:
            train_sampler.set_epoch(train_epoch)
        train_iter = iter(train_loader)
        start_time = time.time()

        progress = tqdm(
            total=round_steps,
            initial=0,
            desc=f"sft:{config.run_name}",
            disable=not distributed.is_master,
        )
        while step < target_step:
            optimizer.zero_grad(set_to_none=True)
            loss_accum = 0.0
            local_tokens_seen = 0
            local_step = step - start_step
            for micro_step in range(config.train.grad_accum_steps):
                try:
                    x, y = next(train_iter)
                except StopIteration:
                    train_epoch += 1
                    if train_sampler is not None:
                        train_sampler.set_epoch(train_epoch)
                    train_iter = iter(train_loader)
                    x, y = next(train_iter)
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                lr = config.train.learning_rate if round_steps <= 1 else (
                    config.train.learning_rate
                    if config.train.min_learning_rate == config.train.learning_rate
                    else config.train.learning_rate
                    - (config.train.learning_rate - config.train.min_learning_rate)
                    * max(local_step - config.train.warmup_steps, 0)
                    / max(round_steps - config.train.warmup_steps, 1)
                )
                for group in optimizer.param_groups:
                    group["lr"] = lr
                sync_context = (
                    model.no_sync if distributed.is_distributed and micro_step < config.train.grad_accum_steps - 1 else contextlib.nullcontext
                )
                with sync_context():
                    with amp_context():
                        _, loss = model(x, y)
                        loss = loss / config.train.grad_accum_steps
                    scaler.scale(loss).backward()
                loss_accum += loss.item()
                local_tokens_seen += int((y != -100).sum().item())
                if micro_step == config.train.grad_accum_steps - 1 and config.train.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)

            scaler.step(optimizer)
            scaler.update()
            step += 1
            global_loss = _reduce_mean(loss_accum, device, distributed)
            tokens_seen += _reduce_sum(local_tokens_seen, device, distributed)
            progress.update(1)

            if distributed.is_master:
                append_csv(
                    train_csv,
                    ["step", "loss", "lr", "tokens_seen"],
                    {"step": step, "loss": global_loss, "lr": lr, "tokens_seen": tokens_seen},
                )
                if step % config.train.log_interval == 0 or step == 1:
                    progress.set_postfix(loss=f"{global_loss:.4f}", lr=f"{lr:.2e}")

            if step % config.train.eval_interval == 0 or step == target_step:
                train_eval = estimate_sft_loss(model, train_loader, device, amp_context, config.train.eval_iters, distributed)
                valid_eval = estimate_sft_loss(model, valid_loader, device, amp_context, config.train.eval_iters, distributed)
                perplexity = math.exp(valid_eval) if valid_eval < 20 else float("inf")

                if distributed.is_master:
                    append_csv(
                        valid_csv,
                        ["step", "train_loss", "valid_loss", "perplexity"],
                        {
                            "step": step,
                            "train_loss": train_eval,
                            "valid_loss": valid_eval,
                            "perplexity": perplexity,
                        },
                    )
                    generated = _unwrap_model(model).generate(
                        prompt_tensor,
                        max_new_tokens=config.generation.max_new_tokens,
                        temperature=config.generation.temperature,
                        top_k=config.generation.top_k,
                        top_p=config.generation.top_p,
                    )
                    text = tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
                    with sample_txt.open("a", encoding="utf-8") as f:
                        f.write(f"[step {step}]\n{text}\n\n")

                    checkpoint = {
                        "step": step,
                        "model_state": raw_model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "config": asdict(config),
                        "best_valid_loss": min(best_val, valid_eval),
                        "tokens_seen": tokens_seen,
                        "resume_from": str(resume_path) if resume_path is not None else None,
                        "world_size": distributed.world_size,
                    }
                    torch.save(checkpoint, ckpt_last)
                    if valid_eval < best_val:
                        best_val = valid_eval
                        torch.save(checkpoint, ckpt_best)

                    save_json(
                        metrics_json,
                        {
                            "run_name": config.run_name,
                            "device": device_type,
                            "dtype": str(amp_dtype),
                            "world_size": distributed.world_size,
                            "params": count_parameters(raw_model),
                            "global_step": step,
                            "session_steps": step - start_step,
                            "train_loss": train_eval,
                            "valid_loss": valid_eval,
                            "perplexity": perplexity,
                            "tokens_seen": tokens_seen,
                            "elapsed_sec": round(time.time() - start_time, 2),
                        },
                    )
        progress.close()

        if not distributed.is_master:
            return None

        return {
            "run_name": config.run_name,
            "params": count_parameters(raw_model),
            "device": device_type,
            "dtype": str(amp_dtype),
            "world_size": distributed.world_size,
            "global_step": step,
            "session_steps": step - start_step,
            "best_valid_loss": best_val,
            "elapsed_sec": round(time.time() - start_time, 2),
        }
    finally:
        _cleanup_distributed(distributed)
