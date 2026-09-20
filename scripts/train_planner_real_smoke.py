#!/usr/bin/env python
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

"""Run a tiny real-data Planning Expert training smoke test.

This script loads the released VLM, freezes it, pre-fills one or more real JSONL
scenes, and trains only the Planning Expert on endpoint flow-matching loss. It is
intended to validate the real data and cache contracts, not to produce a useful model.
Use a CUDA device with enough memory for the selected VLM checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from qwen_drive import QwenDriveForPlanning
from qwen_drive.benchmarks import read_scene_file
from qwen_drive.images import ImageArchive
from qwen_drive.trajectory import normalize_history, normalize_trajectory
from qwen_drive.training import make_flow_batch, masked_endpoint_mse, prefill_frozen_vlm

DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="directory containing the full VLM")
    parser.add_argument(
        "--planner", default=None, help="optional initial Planning Expert directory"
    )
    parser.add_argument("--scenes", required=True, help="planning scene JSONL file")
    parser.add_argument(
        "--image-root",
        default=None,
        help="directory frame paths are relative to (defaults to the scene file directory)",
    )
    parser.add_argument("--image-archive", default=None, help="optional packed frame archive")
    parser.add_argument("--limit", type=int, default=1, help="number of scenes to prefill")
    parser.add_argument("--steps", type=int, default=5, help="optimizer steps")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bfloat16")
    parser.add_argument(
        "--attn-implementation", choices=["sdpa", "flash_attention_2"], default="sdpa"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="optional planner directory; writes model.safetensors and config.json",
    )
    return parser.parse_args()


def move_inputs(inputs: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }


def prepare_samples(args: argparse.Namespace, model, device: torch.device) -> list[dict]:
    archive = ImageArchive.open(args.image_archive) if args.image_archive else None
    image_root = args.image_root
    if image_root is None and archive is None:
        image_root = str(Path(args.scenes).resolve().parent)
    samples = list(
        read_scene_file(
            args.scenes,
            image_root=image_root,
            image_archive=archive,
            num_history_points=model.config.num_history_points,
            limit=args.limit,
        )
    )
    if not samples:
        raise ValueError("scene file did not contain any samples")

    scale = model.trajectory_scale(device)
    prepared = []
    for sample in samples:
        if sample.future_trajectory is None or sample.future_valid is None:
            raise ValueError(f"scene {sample.token!r} has no future trajectory or validity mask")
        future = torch.as_tensor(sample.future_trajectory, dtype=torch.float32, device=device)
        future = torch.nan_to_num(future, nan=0.0, posinf=0.0, neginf=0.0)
        valid = torch.as_tensor(sample.future_valid, dtype=torch.float32, device=device)
        valid = torch.nan_to_num(valid, nan=0.0, posinf=0.0, neginf=0.0).gt(0).float()
        if future.shape != (model.config.num_future_points, model.config.trajectory_point_dim):
            raise ValueError(
                f"scene {sample.token!r} future shape {tuple(future.shape)} does not match "
                f"{model.config.num_future_points, model.config.trajectory_point_dim}"
            )
        if valid.shape != (model.config.num_future_points,):
            raise ValueError(
                f"scene {sample.token!r} validity shape {tuple(valid.shape)} does not match "
                f"({model.config.num_future_points},)"
            )
        if not bool(valid.any()):
            raise ValueError(f"scene {sample.token!r} has no valid future waypoints")

        inputs = move_inputs(model.processor(sample.scene, device="cpu"), device)
        scene_cache, anchor = prefill_frozen_vlm(model, inputs)
        target = normalize_trajectory(future.unsqueeze(0), scale)
        history = normalize_history(inputs["history"].float(), scale)
        noisy, flow_time, _ = make_flow_batch(target)
        prepared.append(
            {
                "token": sample.token,
                "scene_cache": scene_cache,
                "anchor": anchor,
                "history": history,
                "history_velocity": inputs["history_velocity"].float(),
                "history_acceleration": inputs["history_acceleration"].float(),
                "nav_command": inputs["nav_command"],
                "ego_status": inputs["ego_status"].float(),
                "target": target,
                "noisy": noisy,
                "flow_time": flow_time,
                "valid": valid.unsqueeze(0),
            }
        )
        print(
            f"prefilled {sample.token or '<untagged>'}: "
            f"{inputs['input_ids'].shape[1]} text tokens, {len(scene_cache)} cache entries",
            flush=True,
        )
    return prepared


def loss_for_sample(expert, sample: dict) -> torch.Tensor:
    history_queries = expert.encode_history(
        sample["history"],
        sample["nav_command"],
        sample["history_velocity"],
        sample["history_acceleration"],
    )
    prediction = expert.predict_endpoint(
        sample["noisy"],
        sample["flow_time"],
        history_queries,
        sample["scene_cache"],
        sample["anchor"],
        sample["nav_command"],
        sample["ego_status"],
    )
    return masked_endpoint_mse(prediction, sample["target"], sample["valid"])


def mean_loss(expert, samples: list[dict]) -> torch.Tensor:
    return torch.stack([loss_for_sample(expert, sample) for sample in samples]).mean()


def save_planner(model, output: str | Path) -> None:
    from safetensors.torch import save_file

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    weights = {
        f"planning_expert.{name}": value.detach().cpu()
        for name, value in model.planning_expert.state_dict().items()
    }
    save_file(weights, str(output / "model.safetensors"))
    config = model.planning_expert.config.to_dict()
    config.update(
        {
            "num_future_points": model.config.num_future_points,
            "num_history_points": model.config.num_history_points,
            "trajectory_point_dim": model.config.trajectory_point_dim,
        }
    )
    with open(output / "config.json", "w") as handle:
        json.dump(config, handle, indent=2)


def main() -> None:
    args = parse_args()
    if args.limit < 1 or args.steps < 1:
        raise SystemExit("--limit and --steps must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; pass --device cpu only for a model that fits in RAM")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model = QwenDriveForPlanning.from_pretrained(
        args.model,
        planner=args.planner,
        dtype=DTYPES[args.dtype],
        attn_implementation=args.attn_implementation,
    ).to(device)
    model.vlm.requires_grad_(False)
    model.vlm.eval()
    model.planning_expert.train()

    samples = prepare_samples(args, model, device)
    optimizer = torch.optim.AdamW(model.planning_expert.parameters(), lr=args.learning_rate)
    with torch.no_grad():
        initial_loss = float(mean_loss(model.planning_expert, samples))

    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        loss = mean_loss(model.planning_expert, samples)
        loss.backward()
        if not any(parameter.grad is not None for parameter in model.planning_expert.parameters()):
            raise RuntimeError("real smoke test produced no Planning Expert gradients")
        if any(parameter.grad is not None for parameter in model.vlm.parameters()):
            raise RuntimeError("frozen VLM unexpectedly received gradients")
        torch.nn.utils.clip_grad_norm_(model.planning_expert.parameters(), 1.0)
        optimizer.step()
        print(f"step {step + 1}/{args.steps}: loss={float(loss):.6f}", flush=True)

    with torch.no_grad():
        final_loss = float(mean_loss(model.planning_expert, samples))
    if not final_loss < initial_loss:
        raise RuntimeError(f"loss did not decrease: {initial_loss:.6f} -> {final_loss:.6f}")
    if args.output:
        save_planner(model, args.output)
    print(f"real training smoke passed: loss {initial_loss:.6f} -> {final_loss:.6f}")


if __name__ == "__main__":
    main()
