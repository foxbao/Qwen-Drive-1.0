# Copyright 2026 Alibaba Group Holding Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU smoke test for the first Planning Expert training step.

This intentionally uses a tiny real ``PlanningExpert`` and synthetic VLM cache
instead of loading the 4B VLM. It verifies gradients, masked endpoint loss,
single-sample overfitting, and checkpoint round-tripping before real data is
connected to the trainer.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from qwen_drive.configuration_qwen_drive import PlanningExpertConfig
from qwen_drive.planning_expert import PlanningExpert
from qwen_drive.training import make_flow_batch, masked_endpoint_mse


def build_toy_expert() -> PlanningExpert:
    """Create a small expert with the same tensor contracts as the released one."""
    config = PlanningExpertConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=16,
        layers_per_kv=2,
        time_embed_dim=32,
        fourier_num_features=8,
        mrope_section=(2, 0, 0),
    )
    return PlanningExpert(
        config,
        num_future_points=50,
        num_history_points=16,
        trajectory_point_dim=3,
        attn_implementation="sdpa",
    )


def make_inputs(device: torch.device) -> dict[str, torch.Tensor | list]:
    """Create one fixed conditioning example and one smooth target trajectory."""
    steps = torch.linspace(0.02, 1.0, 50, device=device)
    target = torch.stack(
        [0.25 * steps, 0.04 * steps.square(), 0.08 * steps], dim=-1
    ).unsqueeze(0)
    history = torch.zeros(1, 15, 3, device=device)
    history[0, :, 0] = torch.linspace(-0.2, 0.0, 15, device=device)
    velocity = torch.zeros(1, 16, 2, device=device)
    acceleration = torch.zeros(1, 16, 2, device=device)
    nav_command = torch.tensor([0], dtype=torch.long, device=device)
    ego_status = torch.zeros(1, 8, device=device)
    anchor = torch.zeros(3, 1, device=device)
    scene_cache = [
        (
            torch.randn(1, 24, 1, 16, device=device),
            torch.randn(1, 24, 1, 16, device=device),
        )
    ]
    valid_mask = torch.ones(1, 50, device=device)
    valid_mask[:, -5:] = 0
    return {
        "target": target,
        "history": history,
        "velocity": velocity,
        "acceleration": acceleration,
        "nav_command": nav_command,
        "ego_status": ego_status,
        "anchor": anchor,
        "scene_cache": scene_cache,
        "valid_mask": valid_mask,
    }


def run_smoke(steps: int, learning_rate: float, seed: int) -> tuple[float, float]:
    torch.manual_seed(seed)
    device = torch.device("cpu")
    expert = build_toy_expert().to(device)
    expert.train()
    inputs = make_inputs(device)
    optimizer = torch.optim.AdamW(expert.parameters(), lr=learning_rate)

    target = inputs["target"]
    initial_noisy, flow_time, _ = make_flow_batch(
        target,
        flow_time=torch.tensor([0.5], device=device),
        noise=torch.zeros_like(target),
    )

    def loss_for_current_model() -> torch.Tensor:
        history_queries = expert.encode_history(
            inputs["history"],
            inputs["nav_command"],
            inputs["velocity"],
            inputs["acceleration"],
        )
        prediction = expert.predict_endpoint(
            initial_noisy,
            flow_time,
            history_queries,
            inputs["scene_cache"],
            inputs["anchor"],
            inputs["nav_command"],
            inputs["ego_status"],
        )
        return masked_endpoint_mse(prediction, target, inputs["valid_mask"])

    with torch.no_grad():
        initial_loss = float(loss_for_current_model())

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_for_current_model()
        loss.backward()
        if not any(parameter.grad is not None for parameter in expert.parameters()):
            raise RuntimeError("smoke test produced no expert gradients")
        torch.nn.utils.clip_grad_norm_(expert.parameters(), 1.0)
        optimizer.step()

    with torch.no_grad():
        final_loss = float(loss_for_current_model())
    if not final_loss < initial_loss:
        raise RuntimeError(f"loss did not decrease: {initial_loss:.6f} -> {final_loss:.6f}")

    with tempfile.TemporaryDirectory(prefix="qwen_drive_smoke_") as directory:
        path = Path(directory) / "expert.pt"
        torch.save(expert.state_dict(), path)
        restored = build_toy_expert().to(device)
        restored.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        restored.eval()
        with torch.no_grad():
            restored_history_queries = restored.encode_history(
                inputs["history"],
                inputs["nav_command"],
                inputs["velocity"],
                inputs["acceleration"],
            )
            restored_loss = float(
                masked_endpoint_mse(
                    restored.predict_endpoint(
                        initial_noisy,
                        flow_time,
                        restored_history_queries,
                        inputs["scene_cache"],
                        inputs["anchor"],
                        inputs["nav_command"],
                        inputs["ego_status"],
                    ),
                    target,
                    inputs["valid_mask"],
                )
            )
        if abs(restored_loss - final_loss) > 1e-7:
            raise RuntimeError("checkpoint round-trip changed the smoke loss")

    return initial_loss, final_loss


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    initial_loss, final_loss = run_smoke(args.steps, args.learning_rate, args.seed)
    print(f"smoke training passed: loss {initial_loss:.6f} -> {final_loss:.6f}")


if __name__ == "__main__":
    main()
