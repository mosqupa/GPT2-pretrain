#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import torch

import _bootstrap  # noqa: F401
from gpt2_pretrain.config import load_config
from gpt2_pretrain.data import format_sft_prompt
from gpt2_pretrain.tokenizer_utils import load_tokenizer
from gpt2_pretrain.trainer import build_model
from gpt2_pretrain.utils import detect_device

"""
python3 scripts/chat_sft.py \
  --checkpoint outputs/gpt2_mini_4090_sft_20260507_173416/checkpoints/best.pt \
  --config configs/gpt2_mini_4090_sft.yaml
"""

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive chat with pretrained/SFT checkpoints.")
    parser.add_argument("--config", default=None, help="Config yaml path. Optional when auto-discovery can infer it.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path. Optional when selecting from discovered models.")
    parser.add_argument("--instruction", default=None, help="Single-turn instruction text")
    parser.add_argument("--input", default="", help="Optional extra input for single-turn mode")
    parser.add_argument("--temperature", type=float, default=None, help="Override temperature")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Override generation length")
    parser.add_argument("--top-k", type=int, default=None, help="Override top-k")
    parser.add_argument("--top-p", type=float, default=None, help="Override top-p")
    parser.add_argument("--history-turns", type=int, default=3, help="How many previous turns to keep as context")
    parser.add_argument("--list-models", action="store_true", help="List discovered models and exit")
    return parser.parse_args()


def infer_config_from_checkpoint(checkpoint_path: Path) -> Path | None:
    run_dir = checkpoint_path.parent.parent
    run_name = run_dir.name
    match = re.match(r"(.+)_\d{8}_\d{6}$", run_name)
    config_stem = match.group(1) if match else run_name
    candidate = Path("configs") / f"{config_stem}.yaml"
    return candidate if candidate.exists() else None


def discover_models() -> list[dict]:
    outputs_dir = Path("outputs")
    if not outputs_dir.exists():
        return []

    models: list[dict] = []
    seen: set[Path] = set()
    for checkpoint in sorted(outputs_dir.glob("**/checkpoints/best.pt"), reverse=True):
        if checkpoint in seen:
            continue
        seen.add(checkpoint)
        run_dir = checkpoint.parent.parent
        metrics_path = run_dir / "metrics.json"
        metrics = {}
        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                metrics = {}
        config_path = infer_config_from_checkpoint(checkpoint)
        models.append(
            {
                "name": run_dir.name,
                "checkpoint": checkpoint,
                "config": config_path,
                "valid_loss": metrics.get("valid_loss"),
                "global_step": metrics.get("global_step", metrics.get("step")),
                "dtype": metrics.get("dtype"),
            }
        )
    return models


def print_models(models: list[dict]) -> None:
    if not models:
        print("未发现可用模型。")
        return
    print("可选模型：")
    for idx, item in enumerate(models, start=1):
        step = item["global_step"]
        valid_loss = item["valid_loss"]
        config_name = item["config"].name if item["config"] else "未识别配置"
        print(
            f"[{idx}] {item['name']} | config={config_name} | "
            f"step={step if step is not None else '-'} | valid_loss={valid_loss if valid_loss is not None else '-'}"
        )


def resolve_model_selection(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        config_path = Path(args.config) if args.config else infer_config_from_checkpoint(checkpoint_path)
        if config_path is None or not config_path.exists():
            raise FileNotFoundError("无法根据 checkpoint 自动推断配置文件，请显式传入 --config。")
        return config_path, checkpoint_path

    models = discover_models()
    if args.list_models:
        print_models(models)
        raise SystemExit(0)
    if not models:
        raise FileNotFoundError("未在 outputs/ 下发现可用 checkpoint。")

    print_models(models)
    while True:
        raw = input("请选择模型编号（直接回车默认第1个）：").strip()
        if not raw:
            choice = 1
            break
        if raw.isdigit() and 1 <= int(raw) <= len(models):
            choice = int(raw)
            break
        print("输入无效，请重新输入。")

    selected = models[choice - 1]
    if selected["config"] is None:
        raise FileNotFoundError(f"无法为模型 {selected['name']} 自动推断配置文件。")
    return selected["config"], selected["checkpoint"]


def load_chat_model(config_path: Path, checkpoint_path: Path):
    config = load_config(config_path)
    device = detect_device(config.train.device)
    tokenizer = load_tokenizer(config.data.tokenizer_path)
    if tokenizer.get_vocab_size() != config.model.vocab_size:
        config.model.vocab_size = tokenizer.get_vocab_size()
    model = build_model(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return config, device, tokenizer, model


def build_chat_prompt(config, instruction: str, extra_input: str, history: list[tuple[str, str]]) -> str:
    if config.sft is not None:
        history = history[-max(0, args_global.history_turns) :]
        history_text = ""
        if history:
            history_lines = ["以下是先前对话："]
            for user_text, assistant_text in history:
                history_lines.append(f"用户：{user_text}")
                history_lines.append(f"助手：{assistant_text}")
            history_text = "\n".join(history_lines)
        merged_input = extra_input.strip()
        if history_text:
            merged_input = f"{history_text}\n\n{merged_input}".strip() if merged_input else history_text
        return format_sft_prompt(
            instruction,
            merged_input,
            config.sft.system_prompt if config.sft is not None else "",
        )
    return instruction


def generate_reply(config, device, tokenizer, model, prompt: str, args: argparse.Namespace) -> str:
    bos_id = tokenizer.token_to_id("[BOS]")
    prompt_ids = [bos_id] + tokenizer.encode(prompt, add_special_tokens=False).ids
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    generated = model.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens or config.generation.max_new_tokens,
        temperature=args.temperature if args.temperature is not None else config.generation.temperature,
        top_k=args.top_k if args.top_k is not None else config.generation.top_k,
        top_p=args.top_p if args.top_p is not None else config.generation.top_p,
    )
    new_ids = generated[0].tolist()[len(prompt_ids) :]
    text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return text if text else "[空回复]"


def interactive_chat(args: argparse.Namespace) -> None:
    config_path, checkpoint_path = resolve_model_selection(args)
    config, device, tokenizer, model = load_chat_model(config_path, checkpoint_path)
    history: list[tuple[str, str]] = []

    print(f"已加载模型：{checkpoint_path.parent.parent.name}")
    print("输入 /help 查看命令。")

    while True:
        user_text = input("\n你> ").strip()
        if not user_text:
            continue
        if user_text in {"/exit", "/quit"}:
            print("已退出。")
            break
        if user_text == "/help":
            print("/models 查看模型列表")
            print("/switch 切换模型")
            print("/reset 清空对话历史")
            print("/exit 退出")
            continue
        if user_text == "/models":
            print_models(discover_models())
            continue
        if user_text == "/reset":
            history.clear()
            print("对话历史已清空。")
            continue
        if user_text == "/switch":
            config_path, checkpoint_path = resolve_model_selection(
                argparse.Namespace(
                    checkpoint=None,
                    config=None,
                    list_models=False,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    history_turns=args.history_turns,
                    instruction=None,
                    input="",
                )
            )
            config, device, tokenizer, model = load_chat_model(config_path, checkpoint_path)
            history.clear()
            print(f"已切换到模型：{checkpoint_path.parent.parent.name}")
            continue

        prompt = build_chat_prompt(config, user_text, "", history)
        answer = generate_reply(config, device, tokenizer, model, prompt, args)
        print(f"模型> {answer}")
        history.append((user_text, answer))


args_global: argparse.Namespace


def main() -> None:
    global args_global
    args = parse_args()
    args_global = args
    if args.instruction:
        config_path, checkpoint_path = resolve_model_selection(args)
        config, device, tokenizer, model = load_chat_model(config_path, checkpoint_path)
        prompt = build_chat_prompt(config, args.instruction, args.input, [])
        answer = generate_reply(config, device, tokenizer, model, prompt, args)
        print(answer)
        return
    interactive_chat(args)


if __name__ == "__main__":
    main()
