# Copyright 2026 Alibaba Group Holding Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import unittest
import copy
import tempfile

import torch
from torch import nn

from qwen_drive.lora import add_lora_adapter, load_lora_adapter


class LoRATest(unittest.TestCase):
    def test_adapter_freezes_base_and_receives_gradients(self) -> None:
        try:
            import peft  # noqa: F401
        except ImportError:
            self.skipTest("PEFT is an optional dependency")

        class TinyModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_proj = nn.Linear(4, 4)
                self.output = nn.Linear(4, 1)

            def forward(self, value: torch.Tensor) -> torch.Tensor:
                return self.output(self.q_proj(value))

        base = TinyModel()
        initial_state = copy.deepcopy(base.state_dict())
        model = add_lora_adapter(base, rank=2, alpha=4, dropout=0.0, target_modules=("q_proj",))
        model(torch.randn(2, 4)).sum().backward()
        trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
        frozen = [parameter for parameter in model.parameters() if not parameter.requires_grad]
        self.assertTrue(trainable)
        self.assertTrue(all("lora_" in name and parameter.grad is not None for name, parameter in trainable))
        self.assertTrue(frozen)
        self.assertTrue(all(parameter.grad is None for parameter in frozen))

        with tempfile.TemporaryDirectory() as temporary:
            model.save_pretrained(temporary)
            restored_base = TinyModel()
            restored_base.load_state_dict(initial_state)
            restored = load_lora_adapter(restored_base, temporary)
            value = torch.randn(2, 4)
            self.assertTrue(torch.allclose(model.eval()(value), restored.eval()(value)))


if __name__ == "__main__":
    unittest.main()
