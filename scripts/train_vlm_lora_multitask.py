#!/usr/bin/env python
# Copyright 2026 Alibaba Group Holding Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Train a shared VLM LoRA on NAVSIM planning, DriveLM QA and A-OKVQA.

Microbatches are task-homogeneous and sampled by configurable task weights. NAVSIM updates
the LoRA and, when enabled, the Planning Expert; QA tasks update only the LoRA. This first
implementation uses one scene/question at a time to avoid padding variable-size multimodal
inputs.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    try:
        import tomli as tomllib
    except ModuleNotFoundError:  # pragma: no cover - depends on the environment
        tomllib = None

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from qwen_drive import QwenDriveForPlanning
from qwen_drive.benchmarks import BenchmarkSample, read_scene_file
from qwen_drive.lora import DEFAULT_TARGET_MODULES, add_lora_adapter
from qwen_drive.multitask_data import AOKVQAData, DriveLMData, VQASample
from qwen_drive.training import make_flow_batch, masked_endpoint_mse
from qwen_drive.trajectory import normalize_history, normalize_trajectory

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
TASKS = ("navsim", "drivelm", "aokvqa")


def load_config(path: Path) -> dict:
    if tomllib is None:
        raise RuntimeError("TOML config requires Python 3.11+ or the 'tomli' package")
    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    for section in ("training", "data", "lora", "task_weights", "loss_weights"):
        if section not in payload:
            raise ValueError(f"config is missing [{section}]")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/vlm_lora_multitask.toml")
    parser.add_argument("--max-steps", type=int, default=None, help="optimizer updates")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--navsim-train-limit", type=int, default=None)
    parser.add_argument("--navsim-val-limit", type=int, default=None)
    parser.add_argument("--val-samples-per-task", type=int, default=None)
    parser.add_argument(
        "--task-sequence",
        default=None,
        help="debug override, e.g. navsim,drivelm,aokvqa; cycles in this exact order",
    )
    parser.add_argument(
        "--no-save-checkpoint",
        action="store_true",
        help="run the optimization smoke test but do not write a multi-GB planner checkpoint",
    )
    args = parser.parse_args()
    args.config = args.config.expanduser().resolve()
    args.config_payload = load_config(args.config)
    training = args.config_payload["training"]
    args.max_steps = args.max_steps if args.max_steps is not None else int(training["max_steps"])
    args.output = (args.output or Path(training["output"])).expanduser()
    args.navsim_train_limit = (
        args.navsim_train_limit
        if args.navsim_train_limit is not None
        else training.get("navsim_train_limit")
    )
    args.navsim_val_limit = (
        args.navsim_val_limit
        if args.navsim_val_limit is not None
        else training.get("navsim_val_limit")
    )
    args.val_samples_per_task = (
        args.val_samples_per_task
        if args.val_samples_per_task is not None
        else int(training.get("val_samples_per_task", 16))
    )
    args.seed = int(training.get("seed", 3407))
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if args.val_samples_per_task < 1:
        parser.error("--val-samples-per-task must be positive")
    if args.task_sequence:
        args.task_sequence = tuple(x.strip() for x in args.task_sequence.split(",") if x.strip())
        if not args.task_sequence or any(task not in TASKS for task in args.task_sequence):
            parser.error(f"--task-sequence entries must be drawn from {', '.join(TASKS)}")
    return args


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def move_inputs(inputs: dict, device: torch.device) -> dict:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}


def build_vqa_tensors(
    model,
    sample: VQASample,
    device: torch.device,
    *,
    max_target_tokens: int,
    max_sequence_length: int,
    image_pixel_budget: int | None,
) -> tuple[dict, torch.Tensor]:
    """Encode one QA and return its prompt inputs plus assistant-only target token IDs."""
    processor = model.processor
    inputs = processor.encode_vqa(
        sample.images,
        sample.question,
        image_pixel_budget=image_pixel_budget,
        device="cpu",
    )
    answer_ids = processor.tokenizer.encode(sample.answer, add_special_tokens=False)
    answer_ids = answer_ids[:max_target_tokens]
    target_ids = answer_ids + [processor.im_end_id] + list(processor.newline_ids)
    prompt_ids = inputs["input_ids"]
    sequence_length = prompt_ids.shape[1] + len(target_ids)
    if sequence_length > max_sequence_length:
        raise ValueError(
            f"sample {sample.sample_id} has {sequence_length} text/image tokens, exceeding "
            f"max_sequence_length={max_sequence_length}; reduce image budgets or raise the cap"
        )
    input_ids = torch.cat(
        [prompt_ids, torch.tensor([target_ids], dtype=prompt_ids.dtype)], dim=1
    )
    inputs["input_ids"] = input_ids.to(device)
    inputs = move_inputs(inputs, device)
    return inputs, torch.tensor([target_ids], dtype=torch.long, device=device)


def vqa_loss(
    model,
    sample: VQASample,
    device: torch.device,
    *,
    max_target_tokens: int,
    max_sequence_length: int,
    image_pixel_budget: int | None,
) -> torch.Tensor:
    inputs, target_ids = build_vqa_tensors(
        model,
        sample,
        device,
        max_target_tokens=max_target_tokens,
        max_sequence_length=max_sequence_length,
        image_pixel_budget=image_pixel_budget,
    )
    input_ids = inputs["input_ids"]
    target_length = target_ids.shape[1]
    output = model.vlm(
        input_ids=input_ids,
        pixel_values=inputs["pixel_values"],
        image_grid_thw=inputs["image_grid_thw"],
        mm_token_type_ids=model._modality_ids(input_ids),
        use_cache=False,
        # The causal loss only needs logits that predict assistant answer tokens.
        logits_to_keep=target_length + 1,
    )
    logits = output.logits
    if logits.shape[1] != target_length + 1:
        raise RuntimeError(
            f"expected {target_length + 1} answer logits, got {logits.shape[1]}"
        )
    answer_logits = logits[:, :-1, :].float().contiguous()
    return F.cross_entropy(
        answer_logits.view(-1, answer_logits.shape[-1]), target_ids.view(-1)
    )


def navsim_loss(
    model,
    sample: BenchmarkSample,
    device: torch.device,
    *,
    training: bool,
) -> torch.Tensor:
    """Differentiable VLM prefill and Planning Expert endpoint loss for one scene."""
    if sample.future_trajectory is None or sample.future_valid is None:
        raise ValueError(f"NAVSIM scene {sample.token} has no future trajectory/mask")
    inputs = move_inputs(model.processor(sample.scene, device="cpu"), device)
    input_ids = inputs["input_ids"]
    output = model.vlm(
        input_ids=input_ids,
        pixel_values=inputs["pixel_values"],
        image_grid_thw=inputs["image_grid_thw"],
        mm_token_type_ids=model._modality_ids(input_ids),
        use_cache=True,
        logits_to_keep=1,
    )
    if output.past_key_values is None:
        raise RuntimeError("VLM did not return a KV cache for the NAVSIM planning task")
    scene_cache = model._scene_cache(output.past_key_values)
    if training and not any(
        key.requires_grad or value.requires_grad for key, value in scene_cache
    ):
        raise RuntimeError("NAVSIM cache is detached; planner loss cannot update the VLM LoRA")
    anchor = model._rope_positions(input_ids, inputs["image_grid_thw"])[:, :, -1]

    future = torch.as_tensor(sample.future_trajectory, dtype=torch.float32, device=device)
    future = torch.nan_to_num(future, nan=0.0, posinf=0.0, neginf=0.0)
    valid = torch.as_tensor(sample.future_valid, dtype=torch.float32, device=device)
    valid = torch.nan_to_num(valid, nan=0.0, posinf=0.0, neginf=0.0).gt(0).float()
    target = normalize_trajectory(future.unsqueeze(0), model.trajectory_scale(device))
    history = normalize_history(inputs["history"].float(), model.trajectory_scale(device))
    if training:
        noisy, flow_time, _ = make_flow_batch(target)
    else:
        noisy, flow_time, _ = make_flow_batch(
            target,
            flow_time=torch.full((1,), 0.5, device=device),
            noise=torch.zeros_like(target),
        )
    expert = model.planning_expert
    history_queries = expert.encode_history(
        history,
        inputs["nav_command"],
        inputs["history_velocity"].float(),
        inputs["history_acceleration"].float(),
    )
    prediction = expert.predict_endpoint(
        noisy,
        flow_time,
        history_queries,
        scene_cache,
        anchor,
        inputs["nav_command"],
        inputs["ego_status"].float(),
    )
    return masked_endpoint_mse(prediction, target, valid.unsqueeze(0))


def save_checkpoint(model, optimizer, output: Path, state: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    adapter_dir = output / "lora_adapter"
    model.vlm.save_pretrained(adapter_dir)
    save_file(
        {
            f"planning_expert.{name}": value.detach().cpu()
            for name, value in model.planning_expert.state_dict().items()
        },
        str(output / "model.safetensors"),
    )
    with (output / "config.json").open("w") as handle:
        json.dump(model.planning_expert.config.to_dict(), handle, indent=2)
    torch.save({"optimizer": optimizer.state_dict(), **state}, output / "trainer_state.pt")


def main() -> None:
    args = parse_args()
    config = args.config_payload
    training_cfg = config["training"]
    data_cfg = config["data"]
    lora_cfg = config["lora"]
    task_weights = config["task_weights"]
    loss_weights = config["loss_weights"]

    for task in TASKS:
        if float(task_weights.get(task, 0.0)) < 0:
            raise ValueError(f"task weight for {task} cannot be negative")
        if float(loss_weights.get(task, 0.0)) <= 0:
            raise ValueError(f"loss weight for {task} must be positive")
    task_names = [task for task in TASKS if float(task_weights.get(task, 0.0)) > 0]
    if not task_names:
        raise ValueError("at least one task weight must be positive")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device(training_cfg.get("device", "cuda:0"))
    dtype_name = training_cfg.get("dtype", "bfloat16")
    if dtype_name not in DTYPES:
        raise ValueError(f"unsupported dtype {dtype_name!r}")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    model_path = resolve_path(training_cfg["model"])
    planner_path = resolve_path(training_cfg["planner"])
    print(f"loading Qwen-Drive base model: {model_path}", flush=True)
    model = QwenDriveForPlanning.from_pretrained(
        model_path,
        planner=planner_path,
        dtype=DTYPES[dtype_name],
        attn_implementation=training_cfg.get("attn_implementation", "sdpa"),
    ).to(device)
    model.vlm = add_lora_adapter(
        model.vlm,
        rank=int(lora_cfg.get("rank", 8)),
        alpha=int(lora_cfg.get("alpha", 16)),
        dropout=float(lora_cfg.get("dropout", 0.05)),
        target_modules=lora_cfg.get("target_modules", list(DEFAULT_TARGET_MODULES)),
    )
    model.vlm.to(device).train()
    train_planner = bool(training_cfg.get("train_planner", True))
    for parameter in model.planning_expert.parameters():
        parameter.requires_grad_(train_planner)
    if train_planner:
        model.planning_expert.train()
    else:
        model.planning_expert.eval()

    adapter_parameters = [p for p in model.vlm.parameters() if p.requires_grad]
    planner_parameters = [p for p in model.planning_expert.parameters() if p.requires_grad]
    if not adapter_parameters:
        raise RuntimeError("no trainable LoRA parameters were created")
    adapter_count = sum(parameter.numel() for parameter in adapter_parameters)
    base_trainable = [
        name for name, parameter in model.vlm.named_parameters()
        if parameter.requires_grad and "lora_" not in name
    ]
    if base_trainable:
        raise RuntimeError(f"unexpected non-LoRA VLM trainable parameters: {base_trainable[:5]}")

    print("loading NAVSIM train/validation scenes", flush=True)
    navsim_train = list(
        read_scene_file(
            resolve_path(data_cfg["navsim_train"]),
            image_root=resolve_path(data_cfg["navsim_image_root"]),
            num_history_points=model.config.num_history_points,
            limit=args.navsim_train_limit,
        )
    )
    navsim_val = list(
        read_scene_file(
            resolve_path(data_cfg["navsim_val"]),
            image_root=resolve_path(data_cfg["navsim_image_root"]),
            num_history_points=model.config.num_history_points,
            limit=args.navsim_val_limit,
        )
    )
    if not navsim_train or not navsim_val:
        raise ValueError("NAVSIM train and validation scene lists must be non-empty")

    drivelm_train = drivelm_val = None
    aok_train = aok_val = None
    if float(task_weights.get("drivelm", 0.0)) > 0:
        print("loading DriveLM annotations and checking referenced camera files", flush=True)
        drivelm_train = DriveLMData(
            resolve_path(data_cfg["drivelm_annotations"]),
            split="train",
            validation_fraction=float(data_cfg.get("drivelm_validation_fraction", 0.05)),
        )
        drivelm_val = DriveLMData(
            resolve_path(data_cfg["drivelm_annotations"]),
            split="validation",
            validation_fraction=float(data_cfg.get("drivelm_validation_fraction", 0.05)),
        )
    if float(task_weights.get("aokvqa", 0.0)) > 0:
        print("loading A-OKVQA annotations and checking image files", flush=True)
        aok_train = AOKVQAData(
            resolve_path(data_cfg["aok_train_annotations"]),
            resolve_path(data_cfg["aok_train_images"]),
            split="train",
        )
        aok_val = AOKVQAData(
            resolve_path(data_cfg["aok_val_annotations"]),
            resolve_path(data_cfg["aok_val_images"]),
            split="validation",
        )
    print(
        f"samples: NAVSIM {len(navsim_train)} train/{len(navsim_val)} val; "
        f"DriveLM {len(drivelm_train) if drivelm_train is not None else 'disabled'} train/"
        f"{len(drivelm_val) if drivelm_val is not None else 'disabled'} held-out QA; "
        f"A-OKVQA {len(aok_train) if aok_train is not None else 'disabled'} train/"
        f"{len(aok_val) if aok_val is not None else 'disabled'} val",
        flush=True,
    )
    planner_count = sum(p.numel() for p in model.planning_expert.parameters())
    print(
        f"training mode: VLM LoRA + {'Planning Expert' if train_planner else 'frozen Planning Expert'}",
        flush=True,
    )
    print(
        f"LoRA trainable parameters: {adapter_count:,}; "
        f"Planning Expert parameters: {planner_count:,} "
        f"({len(planner_parameters)} tensors trainable)",
        flush=True,
    )

    parameter_groups = [{"params": adapter_parameters, "lr": float(training_cfg["learning_rate"])}]
    if planner_parameters:
        parameter_groups.append(
            {"params": planner_parameters, "lr": float(training_cfg["planner_learning_rate"])}
        )
    optimizer = torch.optim.AdamW(
        parameter_groups,
        weight_decay=float(training_cfg.get("weight_decay", 0.01)),
    )
    max_target_tokens = int(training_cfg.get("max_target_tokens", 256))
    max_sequence_length = int(training_cfg.get("max_sequence_length", 8192))
    image_pixel_budget = training_cfg.get("vqa_image_pixels")
    if image_pixel_budget is not None:
        image_pixel_budget = int(image_pixel_budget)
    accumulation = int(training_cfg.get("gradient_accumulation_steps", 1))
    if accumulation < 1:
        raise ValueError("gradient_accumulation_steps must be positive")

    datasets = {"navsim": navsim_train}
    if drivelm_train is not None:
        datasets["drivelm"] = drivelm_train
    if aok_train is not None:
        datasets["aokvqa"] = aok_train
    val_rng = random.Random(args.seed + 1)
    val_count = args.val_samples_per_task
    val_samples = {"navsim": navsim_val[:val_count]}
    if drivelm_val is not None:
        val_samples["drivelm"] = [drivelm_val.sample(val_rng) for _ in range(val_count)]
    if aok_val is not None:
        val_samples["aokvqa"] = [aok_val.sample(val_rng) for _ in range(val_count)]
    population = [task for task in task_names if task in datasets]
    weights = [float(task_weights[task]) for task in population]
    debug_sequence = args.task_sequence
    if debug_sequence and any(task not in population for task in debug_sequence):
        raise ValueError("--task-sequence includes a task with zero configured sampling weight")

    output = resolve_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "metrics.jsonl"
    update = 0
    progress = tqdm(
        range(args.max_steps),
        desc="VLM LoRA multitask training",
        unit="update",
        dynamic_ncols=True,
        file=sys.stdout,
    )
    for update in progress:
        optimizer.zero_grad(set_to_none=True)
        losses: dict[str, list[float]] = {task: [] for task in TASKS}
        for microstep in range(accumulation):
            task = (
                debug_sequence[(update * accumulation + microstep) % len(debug_sequence)]
                if debug_sequence
                else rng.choices(population, weights=weights, k=1)[0]
            )
            if task == "navsim":
                sample = rng.choice(datasets[task])
                raw_loss = navsim_loss(model, sample, device, training=True)
            else:
                sample = datasets[task].sample(rng)
                raw_loss = vqa_loss(
                    model,
                    sample,
                    device,
                    max_target_tokens=max_target_tokens,
                    max_sequence_length=max_sequence_length,
                    image_pixel_budget=image_pixel_budget,
                )
            scaled_loss = raw_loss * float(loss_weights[task])
            (scaled_loss / accumulation).backward()
            losses[task].append(float(raw_loss.detach()))
            # Release references before constructing the next, potentially larger,
            # multimodal autograd graph.
            del scaled_loss, raw_loss

        if not any(parameter.grad is not None for parameter in adapter_parameters):
            raise RuntimeError("the current task produced no gradient for any VLM LoRA parameter")
        if train_planner and losses["navsim"] and not any(
            parameter.grad is not None for parameter in planner_parameters
        ):
            raise RuntimeError("NAVSIM loss produced no gradient for the Planning Expert")

        if any(parameter.grad is not None for parameter in model.vlm.parameters() if not parameter.requires_grad):
            raise RuntimeError("frozen VLM base unexpectedly received gradients")
        torch.nn.utils.clip_grad_norm_(
            [*adapter_parameters, *planner_parameters], float(training_cfg.get("max_grad_norm", 1.0))
        )
        optimizer.step()
        values = {task: sum(rows) / len(rows) for task, rows in losses.items() if rows}
        progress.set_postfix(
            task="+".join(values),
            loss=" ".join(f"{key}={value:.4g}" for key, value in values.items()),
        )
        update += 1

        eval_every = int(training_cfg.get("eval_every", 0))
        should_evaluate = (eval_every > 0 and update % eval_every == 0) or update == args.max_steps
        if should_evaluate:
            model.vlm.eval()
            model.planning_expert.eval()
            validation: dict[str, float] = {}
            with torch.no_grad():
                for task in population:
                    vals = []
                    for sample in val_samples[task]:
                        if task == "navsim":
                            loss = navsim_loss(model, sample, device, training=False)
                        else:
                            loss = vqa_loss(
                                model,
                                sample,
                                device,
                                max_target_tokens=max_target_tokens,
                                max_sequence_length=max_sequence_length,
                                image_pixel_budget=image_pixel_budget,
                            )
                        vals.append(float(loss))
                    validation[task] = sum(vals) / len(vals)
            row = {"step": update, "train_loss": values, "val_loss": validation}
            with log_path.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            print(f"\nstep {update}: validation losses={validation}", flush=True)
            model.vlm.train()
            if train_planner:
                model.planning_expert.train()
            else:
                model.planning_expert.eval()

    if not args.no_save_checkpoint:
        save_checkpoint(
            model,
            optimizer,
            output,
            {
                "global_step": update,
                "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items() if key != "config_payload"},
                "config": config,
                "trainable_lora_parameters": adapter_count,
                "train_planner": train_planner,
            },
        )
        print(f"saved LoRA adapter and planner checkpoint to {output}", flush=True)
    else:
        print("smoke run complete; checkpoint writing was disabled", flush=True)


if __name__ == "__main__":
    main()
