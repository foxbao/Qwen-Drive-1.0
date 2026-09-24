# Copyright 2026 Alibaba Group Holding Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Optional PEFT LoRA helpers for the Qwen-Drive VLM."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

DEFAULT_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")


def _peft_imports():
    try:
        from peft import LoraConfig, PeftModel, get_peft_model
    except ImportError as error:  # pragma: no cover - depends on optional installation
        raise ImportError(
            "LoRA training/loading requires PEFT; install with "
            "`pip install -e '.[lora]'` or `pip install peft==0.21.0`."
        ) from error
    return LoraConfig, PeftModel, get_peft_model


def add_lora_adapter(
    model,
    *,
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    target_modules: Sequence[str] = DEFAULT_TARGET_MODULES,
):
    """Freeze a VLM and attach a trainable LoRA adapter to selected module suffixes."""
    if rank < 1 or alpha < 1:
        raise ValueError("LoRA rank and alpha must be positive")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("LoRA dropout must be in [0, 1)")
    LoraConfig, _, get_peft_model = _peft_imports()
    model.requires_grad_(False)
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=list(target_modules),
        bias="none",
    )
    adapted = get_peft_model(model, config)
    trainable = [name for name, parameter in adapted.named_parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError(
            "PEFT attached no trainable parameters; verify target_modules against this VLM"
        )
    return adapted


def load_lora_adapter(model, path: str | Path, *, is_trainable: bool = False):
    """Attach a saved PEFT adapter to an already loaded Qwen-Drive VLM."""
    _, PeftModel, _ = _peft_imports()
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"LoRA adapter directory does not exist: {path}")
    return PeftModel.from_pretrained(model, str(path), is_trainable=is_trainable)
