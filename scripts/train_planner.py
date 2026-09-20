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

"""Train only the Qwen-Drive Planning Expert on planning scene JSONL files.

The first trainer intentionally processes one scene at a time.  VLM cache lengths differ
with image sizes and prompt contents, and keeping this constraint explicit makes the first
real trainer easy to inspect.  Use gradient accumulation for a larger effective batch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from qwen_drive import QwenDriveForPlanning
from qwen_drive.benchmarks import BenchmarkSample, read_scene_file
from qwen_drive.images import ImageArchive
from qwen_drive.trajectory import normalize_history, normalize_trajectory
from qwen_drive.training import make_flow_batch, masked_endpoint_mse, prefill_frozen_vlm

DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


class SceneDataset(Dataset):
    """Small wrapper retaining the lazy image references in ``BenchmarkSample``."""

    def __init__(self, samples: list[BenchmarkSample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> BenchmarkSample:
        return self.samples[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="directory containing the full VLM")
    parser.add_argument(
        "--planner", default=None, help="optional initial Planning Expert directory"
    )
    parser.add_argument("--scenes", required=True, help="training scene JSONL file")
    parser.add_argument("--output", required=True, help="planner checkpoint directory")
    parser.add_argument("--val-scenes", default=None, help="optional validation scene JSONL file")
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--image-archive", default=None)
    parser.add_argument("--val-image-root", default=None)
    parser.add_argument("--val-image-archive", default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1, help="must remain 1 for this trainer")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--resume", default=None, help="checkpoint directory to resume")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bfloat16")
    parser.add_argument(
        "--attn-implementation", choices=["sdpa", "flash_attention_2"], default="sdpa"
    )
    return parser.parse_args()


def _image_root(path: str, explicit: str | None, archive: str | None) -> str | None:
    if explicit is not None or archive is not None:
        return explicit
    return str(Path(path).resolve().parent)


def load_samples(
    path: str,
    image_root: str | None,
    image_archive: str | None,
    num_history_points: int,
    limit: int | None,
) -> list[BenchmarkSample]:
    archive = ImageArchive.open(image_archive) if image_archive else None
    samples = list(
        read_scene_file(
            path,
            image_root=_image_root(path, image_root, image_archive),
            image_archive=archive,
            num_history_points=num_history_points,
            limit=limit,
        )
    )
    if not samples:
        raise ValueError(f"no scenes found in {path}")
    return samples


def move_inputs(inputs: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }


def make_training_record(
    sample: BenchmarkSample,
    model,
    device: torch.device,
    training: bool,
) -> dict:
    if sample.future_trajectory is None or sample.future_valid is None:
        raise ValueError(f"scene {sample.token!r} has no future trajectory or validity mask")
    future = torch.as_tensor(sample.future_trajectory, dtype=torch.float32, device=device)
    future = torch.nan_to_num(future, nan=0.0, posinf=0.0, neginf=0.0)
    valid = torch.as_tensor(sample.future_valid, dtype=torch.float32, device=device)
    valid = torch.nan_to_num(valid, nan=0.0, posinf=0.0, neginf=0.0).gt(0).float()
    expected = (model.config.num_future_points, model.config.trajectory_point_dim)
    if tuple(future.shape) != expected or tuple(valid.shape) != (expected[0],):
        raise ValueError(
            f"scene {sample.token!r} has future {tuple(future.shape)} and "
            f"mask {tuple(valid.shape)}; "
            f"expected {expected} and {(expected[0],)}"
        )
    if not bool(valid.any()):
        raise ValueError(f"scene {sample.token!r} has no valid future waypoints")

    inputs = move_inputs(model.processor(sample.scene, device="cpu"), device)
    scene_cache, anchor = prefill_frozen_vlm(model, inputs)
    scale = model.trajectory_scale(device)
    target = normalize_trajectory(future.unsqueeze(0), scale)
    history = normalize_history(inputs["history"].float(), scale)
    if training:
        noisy, flow_time, _ = make_flow_batch(target)
    else:
        noisy, flow_time, _ = make_flow_batch(
            target,
            flow_time=torch.full((1,), 0.5, device=device),
            noise=torch.zeros_like(target),
        )
    return {
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


def record_loss(expert, record: dict) -> torch.Tensor:
    history_queries = expert.encode_history(
        record["history"],
        record["nav_command"],
        record["history_velocity"],
        record["history_acceleration"],
    )
    prediction = expert.predict_endpoint(
        record["noisy"],
        record["flow_time"],
        history_queries,
        record["scene_cache"],
        record["anchor"],
        record["nav_command"],
        record["ego_status"],
    )
    return masked_endpoint_mse(prediction, record["target"], record["valid"])


def save_checkpoint(model, optimizer, scheduler, output: Path, state: dict) -> None:
    from safetensors.torch import save_file

    output.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            f"planning_expert.{name}": value.detach().cpu()
            for name, value in model.planning_expert.state_dict().items()
        },
        str(output / "model.safetensors"),
    )
    with open(output / "config.json", "w") as handle:
        json.dump(model.planning_expert.config.to_dict(), handle, indent=2)
    torch.save(
        {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), **state},
        output / "trainer_state.pt",
    )


def evaluate(model, loader, device: torch.device) -> float:
    losses = []
    with torch.no_grad():
        for sample in loader:
            record = make_training_record(sample, model, device, training=False)
            losses.append(float(record_loss(model.planning_expert, record)))
    return sum(losses) / len(losses)


def main() -> None:
    args = parse_args()
    if args.batch_size != 1:
        raise SystemExit(
            "this first trainer supports --batch-size 1; use accumulation for larger batches"
        )
    if args.epochs < 1 or args.gradient_accumulation_steps < 1:
        raise SystemExit("--epochs and --gradient-accumulation-steps must be positive")
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

    train_samples = load_samples(
        args.scenes,
        args.image_root,
        args.image_archive,
        model.config.num_history_points,
        args.limit,
    )
    val_samples = None
    if args.val_scenes:
        val_samples = load_samples(
            args.val_scenes,
            args.val_image_root,
            args.val_image_archive,
            model.config.num_history_points,
            args.val_limit,
        )
    train_loader = DataLoader(SceneDataset(train_samples), batch_size=1, collate_fn=lambda x: x[0])
    val_loader = (
        None
        if val_samples is None
        else DataLoader(SceneDataset(val_samples), batch_size=1, collate_fn=lambda x: x[0])
    )

    optimizer = torch.optim.AdamW(
        model.planning_expert.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    warmup = args.warmup_steps

    def lr_scale(step: int) -> float:
        if warmup <= 0:
            return 1.0
        return min(1.0, (step + 1) / warmup)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    start_epoch = 0
    global_step = 0
    if args.resume:
        resume = Path(args.resume)
        model.load_planner(resume)
        state_path = resume / "trainer_state.pt"
        if state_path.exists():
            state = torch.load(state_path, map_location=device, weights_only=False)
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            start_epoch = int(state.get("epoch", 0))
            global_step = int(state.get("global_step", 0))
        print(f"resuming from epoch {start_epoch}, update {global_step}", flush=True)

    output = Path(args.output)
    for epoch in range(start_epoch, args.epochs):
        model.planning_expert.train()
        optimizer.zero_grad(set_to_none=True)
        running = []
        for index, sample in enumerate(train_loader):
            record = make_training_record(sample, model, device, training=True)
            loss = record_loss(model.planning_expert, record)
            (loss / args.gradient_accumulation_steps).backward()
            if any(parameter.grad is not None for parameter in model.vlm.parameters()):
                raise RuntimeError("frozen VLM unexpectedly received gradients")
            running.append(float(loss.detach()))
            should_update = (
                (index + 1) % args.gradient_accumulation_steps == 0
                or index + 1 == len(train_loader)
            )
            if should_update:
                torch.nn.utils.clip_grad_norm_(
                    model.planning_expert.parameters(), args.max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
        train_loss = sum(running) / len(running)
        message = f"epoch {epoch + 1}/{args.epochs}: train_loss={train_loss:.6f}"
        if val_loader is not None:
            model.planning_expert.eval()
            val_loss = evaluate(model, val_loader, device)
            message += f" val_loss={val_loss:.6f}"
        print(message, flush=True)
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            output,
            {"epoch": epoch + 1, "global_step": global_step, "args": vars(args)},
        )

    print(f"saved planner checkpoint to {output}")


if __name__ == "__main__":
    main()
