from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    train_bin: str
    valid_bin: str
    tokenizer_path: str
    raw_corpus_path: str
    cleaned_corpus_path: str
    train_split: float
    vocab_size: int
    min_frequency: int
    seq_length: int


@dataclass
class ModelConfig:
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float
    bias: bool = False


@dataclass
class TrainConfig:
    device: str
    dtype: str
    batch_size: int
    grad_accum_steps: int
    num_steps: int
    eval_interval: int
    eval_iters: int
    log_interval: int
    learning_rate: float
    min_learning_rate: float
    warmup_steps: int
    weight_decay: float
    beta1: float
    beta2: float
    grad_clip: float
    resume_from: str | None = None
    num_workers: int = 0


@dataclass
class GenerationConfig:
    max_new_tokens: int
    temperature: float
    top_k: int
    top_p: float


@dataclass
class PathConfig:
    output_dir: str
    checkpoint_dir: str
    sample_dir: str


@dataclass
class ProjectConfig:
    run_name: str
    seed: int
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    generation: GenerationConfig
    paths: PathConfig


def _build_section(section_cls: type[Any], payload: dict[str, Any]) -> Any:
    return section_cls(**payload)


def load_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    
    run_name = raw["run_name"]
    # Provide default paths if not present in YAML
    if "paths" not in raw:
        raw["paths"] = {
            "output_dir": f"outputs/{run_name}",
            "checkpoint_dir": f"outputs/{run_name}/checkpoints",
            "sample_dir": f"outputs/{run_name}/samples",
        }
        
    return ProjectConfig(
        run_name=run_name,
        seed=raw["seed"],
        data=_build_section(DataConfig, raw["data"]),
        model=_build_section(ModelConfig, raw["model"]),
        train=_build_section(TrainConfig, raw["train"]),
        generation=_build_section(GenerationConfig, raw["generation"]),
        paths=_build_section(PathConfig, raw["paths"]),
    )


def ensure_dirs(config: ProjectConfig) -> None:
    for path in (
        config.paths.output_dir,
        config.paths.checkpoint_dir,
        config.paths.sample_dir,
        Path(config.data.train_bin).parent,
        Path(config.data.tokenizer_path).parent,
    ):
        Path(path).mkdir(parents=True, exist_ok=True)
