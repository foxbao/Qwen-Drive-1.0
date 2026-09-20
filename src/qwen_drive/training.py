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

"""Small helpers for Planning Expert training.

The released model uses clean-endpoint flow matching: the expert receives an
interpolated noisy trajectory and predicts the clean target trajectory.  This
module's flow and loss helpers are model-independent. ``prefill_frozen_vlm`` is the
model-aware bridge used by the real trainer: it runs the released VLM once, then
returns the detached cache consumed by the trainable Planning Expert.
"""

from __future__ import annotations

from typing import Optional

import torch

__all__ = ["make_flow_batch", "masked_endpoint_mse", "prefill_frozen_vlm"]


def prefill_frozen_vlm(model, inputs: dict[str, torch.Tensor]) -> tuple[list, torch.Tensor]:
    """Prefill a frozen Qwen-Drive VLM and return detached expert conditions.

    The public inference helpers also run the Planning Expert sampler. Training needs
    only the VLM cache, so this boundary keeps the expert forward differentiable.
    """
    required = ("input_ids", "pixel_values", "image_grid_thw")
    missing = [name for name in required if name not in inputs]
    if missing:
        raise KeyError(f"missing VLM inputs: {', '.join(missing)}")

    input_ids = inputs["input_ids"]
    with torch.no_grad():
        model.vlm.eval()
        outputs = model.vlm(
            input_ids=input_ids,
            pixel_values=inputs["pixel_values"],
            image_grid_thw=inputs["image_grid_thw"],
            mm_token_type_ids=model._modality_ids(input_ids),
            use_cache=True,
        )
        scene_cache = model._scene_cache(outputs.past_key_values)
        position_anchor = model._rope_positions(
            input_ids, inputs["image_grid_thw"]
        )[:, :, -1]

    detached_cache = [(key.detach(), value.detach()) for key, value in scene_cache]
    return detached_cache, position_anchor.detach()


def make_flow_batch(
    target: torch.Tensor,
    *,
    flow_time: Optional[torch.Tensor] = None,
    noise: Optional[torch.Tensor] = None,
    noise_std: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct ``x_t`` for clean-endpoint flow matching.

    Args:
        target: Clean normalized trajectories with shape ``[batch, points, 3]``.
        flow_time: Optional times with shape ``[batch]`` in ``[0, 1]``. If omitted,
            times are sampled uniformly.
        noise: Optional initial Gaussian samples with the same shape as ``target``.
        noise_std: Standard deviation used when ``noise`` is not supplied.

    Returns:
        ``(noisy_trajectory, flow_time, noise)``. The interpolation is
        ``x_t = (1 - t) * noise + t * target``.
    """
    if target.ndim != 3:
        raise ValueError(f"target must have shape [batch, points, dim], got {target.shape}")
    target = target.float()
    batch = target.shape[0]

    if flow_time is None:
        flow_time = torch.rand(batch, device=target.device, dtype=target.dtype)
    else:
        flow_time = flow_time.to(device=target.device, dtype=target.dtype)
        if flow_time.shape != (batch,):
            raise ValueError(f"flow_time must have shape [{batch}], got {flow_time.shape}")
    if torch.any((flow_time < 0) | (flow_time > 1)):
        raise ValueError("flow_time must be within [0, 1]")

    if noise is None:
        noise = torch.randn_like(target) * noise_std
    else:
        noise = noise.to(device=target.device, dtype=target.dtype)
        if noise.shape != target.shape:
            raise ValueError(f"noise must match target shape {target.shape}, got {noise.shape}")

    weight = flow_time.view(batch, 1, 1)
    noisy = torch.lerp(noise, target, weight)
    return noisy, flow_time, noise


def masked_endpoint_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean squared endpoint error over valid future poses only.

    ``valid_mask`` is normally ``[batch, points]``. Samples with no valid points
    contribute a differentiable zero instead of producing a NaN.
    """
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError(
            "prediction and target must have the same [batch, points, dim] shape; "
            f"got {prediction.shape} and {target.shape}"
        )
    if valid_mask.shape == prediction.shape[:-1]:
        valid_mask = valid_mask.unsqueeze(-1)
    if valid_mask.shape != prediction.shape[:2] + (1,):
        raise ValueError(
            f"valid_mask must have shape {prediction.shape[:2]} or "
            f"{prediction.shape[:2] + (1,)}, got {valid_mask.shape}"
        )

    weights = torch.nan_to_num(
        valid_mask.to(device=prediction.device, dtype=prediction.dtype),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    raw_error = (prediction.float() - target.float()).square()
    # Select the masked branch before reduction so NaNs in padded labels cannot leak
    # through ``0 * NaN``. Non-finite values in valid labels remain visible to the caller.
    error = torch.where(weights > 0, raw_error, torch.zeros_like(raw_error))
    valid_coordinates = weights.sum() * prediction.shape[-1]
    if valid_coordinates.item() == 0:
        return prediction.sum() * 0.0
    return (error * weights).sum() / valid_coordinates
