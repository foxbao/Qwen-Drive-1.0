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

"""Compare paired NAVSIM predictions and plot the scenes with the largest regressions.

The default configuration compares both the direct and reasoning NAVSIM runs already
stored under ``outputs/``. ADE/FDE use the same first-candidate, first-4-second metric as
``scripts/eval_navsim.py``. Plots are ego-frame trajectory diagrams, not map renderings.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    try:
        import tomli as tomllib
    except ModuleNotFoundError:  # pragma: no cover - depends on the environment
        tomllib = None

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "navsim_regression_review.toml"
CONFIG_KEYS = {"output", "top_k", "tie_tolerance", "experiments"}


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def load_config(path: Path) -> dict:
    if tomllib is None:
        raise RuntimeError("TOML config requires Python 3.11+ or the 'tomli' package")
    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    payload = payload.get("review", payload)
    unknown = sorted(set(payload) - CONFIG_KEYS)
    if unknown:
        raise ValueError(f"unknown review config key(s): {', '.join(unknown)}")
    if not isinstance(payload.get("experiments"), list) or not payload["experiments"]:
        raise ValueError("config must define at least one [[review.experiments]] entry")
    return payload


def read_jsonl(path: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.endswith("\n"):
                break  # ignore a possibly truncated final record
            row = json.loads(line)
            token = str(row.get("token", ""))
            if not token:
                raise ValueError(f"{path}:{line_number} has no token")
            if token in records:
                raise ValueError(f"duplicate token {token!r} in {path}")
            records[token] = row
    return records


def first_trajectory(record: dict) -> np.ndarray:
    trajectories = np.asarray(record["trajectories"], dtype=np.float64)
    if trajectories.ndim == 2:
        trajectories = trajectories[None]
    if trajectories.ndim != 3 or trajectories.shape[0] == 0 or trajectories.shape[2] < 2:
        raise ValueError(f"invalid trajectories shape: {trajectories.shape}")
    return trajectories[0]


def scene_error(record: dict, gt: np.ndarray | None = None) -> tuple[float, float, np.ndarray, np.ndarray]:
    prediction = first_trajectory(record)
    if gt is None:
        if record.get("future_trajectory") is None:
            raise ValueError("record has no future_trajectory")
        gt = np.asarray(record["future_trajectory"], dtype=np.float64)
    points = min(40, prediction.shape[0], gt.shape[0])
    if points < 1:
        raise ValueError("prediction and ground truth have no overlapping points")
    error = np.linalg.norm(prediction[:points, :2] - gt[:points, :2], axis=-1)
    return float(error.mean()), float(error[-1]), prediction[:points], gt[:points]


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def paired_rows(baseline_path: Path, updated_path: Path, tie_tolerance: float) -> tuple[list[dict], dict]:
    baseline = read_jsonl(baseline_path)
    updated = read_jsonl(updated_path)
    common = sorted(set(baseline) & set(updated))
    if not common:
        raise ValueError(f"no matching tokens between {baseline_path} and {updated_path}")

    rows = []
    for token in common:
        before, after = baseline[token], updated[token]
        if before.get("future_trajectory") is None or after.get("future_trajectory") is None:
            continue
        gt_before = np.asarray(before["future_trajectory"], dtype=np.float64)
        gt_after = np.asarray(after["future_trajectory"], dtype=np.float64)
        if gt_before.shape != gt_after.shape or not np.allclose(
            gt_before, gt_after, equal_nan=True, atol=1e-4
        ):
            raise ValueError(f"ground-truth mismatch for token {token}")
        ade_before, fde_before, _, _ = scene_error(before, gt_before)
        ade_after, fde_after, _, _ = scene_error(after, gt_before)
        pred_before = first_trajectory(before)[:40]
        pred_after = first_trajectory(after)[:40]
        gt_window = gt_before[: min(40, len(pred_before), len(pred_after), len(gt_before))]
        pred_before = pred_before[: len(gt_window)]
        pred_after = pred_after[: len(gt_window)]
        forward_before = pred_before[:, 0] - gt_window[:, 0]
        forward_after = pred_after[:, 0] - gt_window[:, 0]
        lateral_before = pred_before[:, 1] - gt_window[:, 1]
        lateral_after = pred_after[:, 1] - gt_window[:, 1]
        rows.append(
            {
                "token": token,
                "scene_token": str(after.get("scene_token", before.get("scene_token", ""))),
                "ade_baseline": ade_before,
                "ade_updated": ade_after,
                "delta_ade": ade_after - ade_before,
                "fde_baseline": fde_before,
                "fde_updated": fde_after,
                "delta_fde": fde_after - fde_before,
                "signed_forward_error_baseline": float(forward_before.mean()),
                "signed_forward_error_updated": float(forward_after.mean()),
                "delta_abs_forward_mae": float(
                    np.abs(forward_after).mean() - np.abs(forward_before).mean()
                ),
                "delta_abs_lateral_mae": float(
                    np.abs(lateral_after).mean() - np.abs(lateral_before).mean()
                ),
                "valid_points": int(np.asarray(after.get("future_valid", []), dtype=float).sum()),
                "reasoning": after.get("reasoning") or "",
            }
        )

    if not rows:
        raise ValueError("no paired records with ground-truth trajectories")
    rows.sort(key=lambda row: row["delta_ade"], reverse=True)
    ade_improved = sum(row["delta_ade"] < -tie_tolerance for row in rows)
    ade_worsened = sum(row["delta_ade"] > tie_tolerance for row in rows)
    fde_improved = sum(row["delta_fde"] < -tie_tolerance for row in rows)
    fde_worsened = sum(row["delta_fde"] > tie_tolerance for row in rows)
    summary = {
        "num_paired": len(rows),
        "num_baseline_only": len(set(baseline) - set(updated)),
        "num_updated_only": len(set(updated) - set(baseline)),
        "ADE_4s_baseline": float(np.mean([row["ade_baseline"] for row in rows])),
        "ADE_4s_updated": float(np.mean([row["ade_updated"] for row in rows])),
        "ADE_4s_relative_change_percent": 100.0
        * (
            np.mean([row["ade_updated"] for row in rows])
            - np.mean([row["ade_baseline"] for row in rows])
        )
        / max(np.mean([row["ade_baseline"] for row in rows]), 1e-12),
        "FDE_4s_baseline": float(np.mean([row["fde_baseline"] for row in rows])),
        "FDE_4s_updated": float(np.mean([row["fde_updated"] for row in rows])),
        "ADE_improved": ade_improved,
        "ADE_worsened": ade_worsened,
        "ADE_tied": len(rows) - ade_improved - ade_worsened,
        "FDE_improved": fde_improved,
        "FDE_worsened": fde_worsened,
        "FDE_tied": len(rows) - fde_improved - fde_worsened,
        "top_12_mean_delta_abs_forward_mae": float(
            np.mean([row["delta_abs_forward_mae"] for row in rows[:12]])
        ),
        "top_12_mean_delta_abs_lateral_mae": float(
            np.mean([row["delta_abs_lateral_mae"] for row in rows[:12]])
        ),
        "all_scenes_mean_delta_signed_forward_error": float(
            np.mean(
                [
                    row["signed_forward_error_updated"]
                    - row["signed_forward_error_baseline"]
                    for row in rows
                ]
            )
        ),
    }
    return rows, summary


def plot_regression(
    baseline_record: dict,
    updated_record: dict,
    row: dict,
    baseline_name: str,
    updated_name: str,
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ade0, fde0, pred0, gt = scene_error(baseline_record)
    ade1, fde1, pred1, _ = scene_error(updated_record, gt)
    points = min(len(gt), len(pred0), len(pred1))
    gt, pred0, pred1 = gt[:points], pred0[:points], pred1[:points]
    time = np.arange(points) / 10.0
    error0 = np.linalg.norm(pred0[:, :2] - gt[:, :2], axis=-1)
    error1 = np.linalg.norm(pred1[:, :2] - gt[:, :2], axis=-1)

    fig, (trajectory_ax, error_ax) = plt.subplots(
        1, 2, figsize=(12, 6), gridspec_kw={"width_ratios": [1.05, 1.0]}
    )
    # NAVSIM ego frame: x is forward, y is left/right. Put forward on the vertical axis.
    trajectory_ax.plot(gt[:, 1], gt[:, 0], color="black", linewidth=2.4, label="ground truth")
    trajectory_ax.plot(
        pred0[:, 1], pred0[:, 0], color="#2878b5", linewidth=2, label=f"{baseline_name}  ADE {ade0:.2f}m"
    )
    trajectory_ax.plot(
        pred1[:, 1], pred1[:, 0], color="#e67e22", linewidth=2, label=f"{updated_name}  ADE {ade1:.2f}m"
    )
    trajectory_ax.scatter([0], [0], marker="s", s=45, color="#c0392b", zorder=4, label="ego at t=0")
    trajectory_ax.scatter([gt[-1, 1]], [gt[-1, 0]], marker="x", color="black", s=50)
    trajectory_ax.set_xlabel("lateral y [m]  (left positive)")
    trajectory_ax.set_ylabel("longitudinal x [m]  (forward)")
    trajectory_ax.set_title("4-second trajectory (ego frame)")
    trajectory_ax.set_aspect("equal", adjustable="datalim")
    trajectory_ax.grid(alpha=0.25)
    trajectory_ax.legend(fontsize=8, loc="best")

    error_ax.plot(time, error0, color="#2878b5", linewidth=1.8, label=f"{baseline_name}  FDE {fde0:.2f}m")
    error_ax.plot(time, error1, color="#e67e22", linewidth=1.8, label=f"{updated_name}  FDE {fde1:.2f}m")
    error_ax.set_xlabel("time [s]")
    error_ax.set_ylabel("position error [m]")
    error_ax.set_title("Position error to ground truth")
    error_ax.set_xlim(0, max(time[-1], 0.1))
    error_ax.grid(alpha=0.25)
    error_ax.legend(fontsize=8, loc="best")

    reason = row.get("reasoning", "")
    title = (
        f"token={row['token']} | ΔADE={row['delta_ade']:+.3f}m | "
        f"ΔFDE={row['delta_fde']:+.3f}m"
    )
    if reason:
        title += "\nreasoning: " + reason[:150]
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()
    config_path = resolve_path(args.config)
    config = load_config(config_path)
    output_root = resolve_path(args.output_dir or config["output"])
    top_k = args.top_k if args.top_k is not None else int(config.get("top_k", 12))
    tie_tolerance = float(config.get("tie_tolerance", 1e-6))
    if top_k < 1 or tie_tolerance < 0:
        parser.error("top-k must be positive and tie-tolerance must be non-negative")

    output_root.mkdir(parents=True, exist_ok=True)
    for experiment in config["experiments"]:
        name = str(experiment["name"])
        baseline_name = str(experiment.get("baseline_name", "baseline"))
        updated_name = str(experiment.get("updated_name", "updated"))
        baseline_path = resolve_path(experiment["baseline"])
        updated_path = resolve_path(experiment["updated"])
        baseline_records = read_jsonl(baseline_path)
        updated_records = read_jsonl(updated_path)
        rows, summary = paired_rows(baseline_path, updated_path, tie_tolerance)
        destination = output_root / safe_name(name)
        plots = destination / "plots"
        plots.mkdir(parents=True, exist_ok=True)

        csv_path = destination / "per_scene.csv"
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        regressions = [row for row in rows if row["delta_ade"] > tie_tolerance][:top_k]
        for rank, row in enumerate(regressions, 1):
            plot_regression(
                baseline_records[row["token"]],
                updated_records[row["token"]],
                row,
                baseline_name,
                updated_name,
                plots / f"{rank:02d}_{safe_name(row['token'])}.png",
            )

        summary["top_ade_regressions"] = [
            {
                key: row[key]
                for key in (
                    "token",
                    "scene_token",
                    "delta_ade",
                    "delta_fde",
                    "ade_baseline",
                    "ade_updated",
                    "fde_baseline",
                    "fde_updated",
                    "delta_abs_forward_mae",
                    "delta_abs_lateral_mae",
                )
            }
            for row in regressions
        ]
        summary_path = destination / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"[{name}] paired={summary['num_paired']}")
        print(
            f"  ADE {summary['ADE_4s_baseline']:.4f} -> {summary['ADE_4s_updated']:.4f} m; "
            f"improved/worsened/tied={summary['ADE_improved']}/{summary['ADE_worsened']}/{summary['ADE_tied']}"
        )
        print(
            f"  FDE {summary['FDE_4s_baseline']:.4f} -> {summary['FDE_4s_updated']:.4f} m; "
            f"improved/worsened/tied={summary['FDE_improved']}/{summary['FDE_worsened']}/{summary['FDE_tied']}"
        )
        print(
            "  worst-12 mean change in |forward/lateral error|: "
            f"{summary['top_12_mean_delta_abs_forward_mae']:+.3f}/"
            f"{summary['top_12_mean_delta_abs_lateral_mae']:+.3f} m"
        )
        print(f"  worst ADE regressions: {len(regressions)} plots -> {plots}")
        print(f"  per-scene table: {csv_path}")


if __name__ == "__main__":
    main()
