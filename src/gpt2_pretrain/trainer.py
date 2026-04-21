from __future__ import annotations

import contextlib
from dataclasses import asdict
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
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


def make_dataloader(bin_path: str, seq_length: int, batch_size: int, num_workers: int) -> DataLoader:
    dataset = MemmapDataset(bin_path, seq_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def estimate_loss(
    model: GPT,
    data_loader: DataLoader,
    device: str,
    amp_context,
    eval_iters: int,
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
    return sum(losses) / len(losses)


def _make_amp_context(device: str, dtype: torch.dtype):
    if device == "cuda" and dtype in (torch.float16, torch.bfloat16):
        return torch.autocast(device_type="cuda", dtype=dtype)
    return contextlib.nullcontext()


def train(config: ProjectConfig) -> dict:
    ensure_dirs(config)
    set_seed(config.seed)
    device = detect_device(config.train.device)
    amp_dtype = detect_dtype(config.train.dtype, device)
    model = build_model(config).to(device)
    train_loader = make_dataloader(
        config.data.train_bin, config.data.seq_length, config.train.batch_size, config.train.num_workers
    )
    valid_loader = make_dataloader(
        config.data.valid_bin, config.data.seq_length, config.train.batch_size, config.train.num_workers
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        betas=(config.train.beta1, config.train.beta2),
        weight_decay=config.train.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda" and amp_dtype == torch.float16))
    amp_context = lambda: _make_amp_context(device, amp_dtype)

    train_csv = Path(config.paths.output_dir) / "train_loss.csv"
    valid_csv = Path(config.paths.output_dir) / "valid_loss.csv"
    sample_txt = Path(config.paths.sample_dir) / "progress_samples.txt"
    metrics_json = Path(config.paths.output_dir) / "metrics.json"
    ckpt_last = Path(config.paths.checkpoint_dir) / "last.pt"
    ckpt_best = Path(config.paths.checkpoint_dir) / "best.pt"

    tokenizer = load_tokenizer(config.data.tokenizer_path)
    bos_id = tokenizer.token_to_id("[BOS]")
    prompt_ids = torch.tensor([[bos_id]], dtype=torch.long, device=device)

    best_val = float("inf")
    step = 0
    tokens_seen = 0
    train_iter = iter(train_loader)
    start_time = time.time()

    progress = tqdm(total=config.train.num_steps, desc=f"training:{config.run_name}")
    while step < config.train.num_steps:
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for micro_step in range(config.train.grad_accum_steps):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)
            x = x.to(device)
            y = y.to(device)
            lr = cosine_lr(
                step,
                config.train.num_steps,
                config.train.warmup_steps,
                config.train.learning_rate,
                config.train.min_learning_rate,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            with amp_context():
                _, loss = model(x, y)
                loss = loss / config.train.grad_accum_steps
            scaler.scale(loss).backward()
            loss_accum += loss.item()
            tokens_seen += x.numel()
            if micro_step == config.train.grad_accum_steps - 1 and config.train.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)

        scaler.step(optimizer)
        scaler.update()
        step += 1
        progress.update(1)

        append_csv(
            train_csv,
            ["step", "loss", "lr", "tokens_seen"],
            {"step": step, "loss": loss_accum, "lr": lr, "tokens_seen": tokens_seen},
        )

        if step % config.train.log_interval == 0 or step == 1:
            progress.set_postfix(loss=f"{loss_accum:.4f}", lr=f"{lr:.2e}")

        if step % config.train.eval_interval == 0 or step == config.train.num_steps:
            train_eval = estimate_loss(model, train_loader, device, amp_context, config.train.eval_iters)
            valid_eval = estimate_loss(model, valid_loader, device, amp_context, config.train.eval_iters)
            perplexity = math.exp(valid_eval) if valid_eval < 20 else float("inf")
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
            generated = model.generate(
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
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": asdict(config),
                "best_valid_loss": min(best_val, valid_eval),
            }
            torch.save(checkpoint, ckpt_last)
            if valid_eval < best_val:
                best_val = valid_eval
                torch.save(checkpoint, ckpt_best)

            save_json(
                metrics_json,
                {
                    "run_name": config.run_name,
                    "device": device,
                    "dtype": str(amp_dtype),
                    "params": count_parameters(model),
                    "step": step,
                    "train_loss": train_eval,
                    "valid_loss": valid_eval,
                    "perplexity": perplexity,
                    "tokens_seen": tokens_seen,
                    "elapsed_sec": round(time.time() - start_time, 2),
                },
            )
    progress.close()
    return {
        "run_name": config.run_name,
        "params": count_parameters(model),
        "device": device,
        "dtype": str(amp_dtype),
        "best_valid_loss": best_val,
        "elapsed_sec": round(time.time() - start_time, 2),
    }
