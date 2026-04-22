from __future__ import annotations

import contextlib
from dataclasses import asdict, dataclass
from datetime import datetime
import math
import os
import time
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from gpt2_pretrain.config import ProjectConfig, ensure_dirs
from gpt2_pretrain.data import MemmapDataset
from gpt2_pretrain.model import GPT, GPTConfig
from gpt2_pretrain.tokenizer_utils import load_tokenizer
from gpt2_pretrain.utils import (
    append_csv,
    cosine_lr,
    count_parameters,
    detect_device,
    detect_dtype,
    save_json,
    set_seed,
)


def _is_cuda_device(device: str) -> bool:
    return device.startswith("cuda")


@dataclass
class DistributedContext:
    is_distributed: bool
    rank: int
    local_rank: int
    world_size: int
    is_master: bool
    device: str


def build_model(config: ProjectConfig) -> GPT:
    model_config = GPTConfig(
        vocab_size=config.model.vocab_size,
        block_size=config.data.seq_length,
        n_layer=config.model.n_layer,
        n_head=config.model.n_head,
        n_embd=config.model.n_embd,
        dropout=config.model.dropout,
        bias=config.model.bias,
    )
    return GPT(model_config)


def make_dataloader(
    bin_path: str,
    seq_length: int,
    batch_size: int,
    num_workers: int,
    distributed: DistributedContext,
    shuffle: bool,
) -> tuple[DataLoader, DistributedSampler | None]:
    dataset = MemmapDataset(bin_path, seq_length)
    sampler = None
    if distributed.is_distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=distributed.world_size,
            rank=distributed.rank,
            shuffle=shuffle,
            drop_last=True,
        )
    return (
        DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle and sampler is None,
            sampler=sampler,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=_is_cuda_device(distributed.device),
        ),
        sampler,
    )


def _setup_distributed(requested_device: str) -> DistributedContext:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_distributed = world_size > 1
    if is_distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend=backend)
        detected_device = detect_device(requested_device)
        if _is_cuda_device(detected_device):
            torch.cuda.set_device(local_rank)
            device = f"cuda:{local_rank}"
        else:
            device = "cpu"
    else:
        device = detect_device(requested_device)
    return DistributedContext(
        is_distributed=is_distributed,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        is_master=(rank == 0),
        device=device,
    )


def _cleanup_distributed(distributed: DistributedContext) -> None:
    if distributed.is_distributed and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def _reduce_mean(value: float, device: str, distributed: DistributedContext) -> float:
    if not distributed.is_distributed:
        return value
    tensor = torch.tensor(value, device=device, dtype=torch.float64)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor /= distributed.world_size
    return tensor.item()


def _reduce_sum(value: int, device: str, distributed: DistributedContext) -> int:
    if not distributed.is_distributed:
        return value
    tensor = torch.tensor(value, device=device, dtype=torch.long)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return int(tensor.item())


def _unwrap_model(model: GPT | DistributedDataParallel) -> GPT:
    if isinstance(model, DistributedDataParallel):
        return model.module
    return model


@torch.no_grad()
def estimate_loss(
    model: GPT | DistributedDataParallel,
    data_loader: DataLoader,
    device: str,
    amp_context,
    eval_iters: int,
    distributed: DistributedContext,
) -> float:
    model.eval()
    losses = []
    data_iter = iter(data_loader)
    for _ in range(eval_iters):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            x, y = next(data_iter)
        x = x.to(device)
        y = y.to(device)
        with amp_context():
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    loss_value = sum(losses) / len(losses)
    return _reduce_mean(loss_value, device, distributed)


def _make_amp_context(device: str, dtype: torch.dtype):
    if _is_cuda_device(device) and dtype in (torch.float16, torch.bfloat16):
        return torch.autocast(device_type="cuda", dtype=dtype)
    return contextlib.nullcontext()


def _resolve_resume_path(config: ProjectConfig) -> Path | None:
    if not config.train.resume_from:
        return None
    return Path(config.train.resume_from)


def _load_checkpoint(
    checkpoint_path: Path,
    raw_model: GPT,
    optimizer: torch.optim.Optimizer,
    device: str,
    distributed: DistributedContext,
) -> tuple[int, float, int]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    raw_model.load_state_dict(checkpoint["model_state"])

    optimizer_state = checkpoint.get("optimizer_state")
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    step = int(checkpoint.get("step", 0))
    best_valid_loss = float(checkpoint.get("best_valid_loss", float("inf")))
    tokens_seen = int(checkpoint.get("tokens_seen", 0))

    if distributed.is_master:
        print(f"[resume] loaded checkpoint from {checkpoint_path} at step={step}")

    return step, best_valid_loss, tokens_seen


def train(config: ProjectConfig) -> dict | None:
    # Generate dynamic output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("outputs") / f"{config.run_name}_{timestamp}"
    
    # Update config paths to use the new timestamped directory
    config.paths.output_dir = str(run_dir)
    config.paths.checkpoint_dir = str(run_dir / "checkpoints")
    config.paths.sample_dir = str(run_dir / "samples")
    
    ensure_dirs(config)
    distributed = _setup_distributed(config.train.device)
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

        train_loader, train_sampler = make_dataloader(
            config.data.train_bin,
            config.data.seq_length,
            config.train.batch_size,
            config.train.num_workers,
            distributed,
            shuffle=True,
        )
        valid_loader, _ = make_dataloader(
            config.data.valid_bin,
            config.data.seq_length,
            config.train.batch_size,
            config.train.num_workers,
            distributed,
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
        prompt_ids = torch.tensor([[bos_id]], dtype=torch.long, device=device)

        best_val = float("inf")
        step = 0
        tokens_seen = 0
        start_step = 0
        resume_path = _resolve_resume_path(config)
        if resume_path is not None:
            if not resume_path.exists():
                raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
            step, best_val, tokens_seen = _load_checkpoint(
                resume_path,
                raw_model,
                optimizer,
                device,
                distributed,
            )
            start_step = step
            if step >= config.train.num_steps:
                raise ValueError(
                    f"Resume checkpoint step={step} has already reached train.num_steps={config.train.num_steps}. "
                    "Please increase num_steps in the config to continue training."
                )

        train_epoch = 0
        if train_sampler is not None:
            train_sampler.set_epoch(train_epoch)
        train_iter = iter(train_loader)
        start_time = time.time()

        progress = tqdm(
            total=config.train.num_steps,
            initial=step,
            desc=f"training:{config.run_name}",
            disable=not distributed.is_master,
        )
        while step < config.train.num_steps:
            optimizer.zero_grad(set_to_none=True)
            loss_accum = 0.0
            local_tokens_seen = 0
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
                lr = cosine_lr(
                    step,
                    config.train.num_steps,
                    config.train.warmup_steps,
                    config.train.learning_rate,
                    config.train.min_learning_rate,
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
                local_tokens_seen += x.numel()
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

            if step % config.train.eval_interval == 0 or step == config.train.num_steps:
                train_eval = estimate_loss(model, train_loader, device, amp_context, config.train.eval_iters, distributed)
                valid_eval = estimate_loss(model, valid_loader, device, amp_context, config.train.eval_iters, distributed)
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
                        prompt_ids,
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
                            "timestamp": timestamp,
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
            "timestamp": timestamp,
            "params": count_parameters(raw_model),
            "device": device_type,
            "dtype": str(amp_dtype),
            "world_size": distributed.world_size,
            "global_step": step,
            "best_valid_loss": best_val,
            "elapsed_sec": round(time.time() - start_time, 2),
        }

    finally:
        _cleanup_distributed(distributed)
