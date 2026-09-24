#!/usr/bin/env python3
"""Convert NAVSIM/OpenScene trainval frames into Qwen-Drive scene JSONL.

The downloaded NAVSIM metadata is sampled at 2 Hz (0.5 s).  NAVSIM's default
scene window is four history frames plus ten future frames.  This converter uses
that same window, resolves the current images from ``navtrain_current_*`` and the
three history images from ``navtrain_history_*``, then linearly resamples the
metadata poses and ego state to Qwen-Drive's 10 Hz contract.

This is deliberately a first, inspectable converter.  It does not copy image
bytes.  The produced image paths remain relative to ``--data-root`` and therefore
work with Qwen-Drive's ``--image-root`` option.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Iterable

import numpy as np


CAMERAS = ("CAM_F0", "CAM_L0", "CAM_R0")
VIEW_LABELS = ("<FRONT VIEW>", "<FRONT LEFT VIEW>", "<FRONT RIGHT VIEW>")
NAV_NAMES = ("GO STRAIGHT", "TURN LEFT", "TURN RIGHT")
WINDOW_HISTORY = 4
WINDOW_FUTURE = 10
WINDOW_SIZE = WINDOW_HISTORY + WINDOW_FUTURE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--include-unknown",
        action="store_true",
        help="keep rows whose driving command is unknown (mapped to GO STRAIGHT)",
    )
    parser.add_argument(
        "--allow-no-route",
        action="store_true",
        help="do not require non-empty roadblock_ids on the current frame",
    )
    parser.add_argument(
        "--max-pkls",
        type=int,
        default=None,
        help="only read the first N metadata files (smoke testing)",
    )
    return parser.parse_args()


def wrap_angle(angle: np.ndarray | float) -> np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))


def quaternion_yaw(quaternion: Any) -> float:
    """Yaw for a [w, x, y, z] quaternion, matching pyquaternion."""
    w, x, y, z = (float(value) for value in quaternion)
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def global_pose(row: dict[str, Any]) -> np.ndarray:
    translation = np.asarray(row["ego2global_translation"], dtype=np.float64)
    return np.array([translation[0], translation[1], quaternion_yaw(row["ego2global_rotation"])])


def to_current_frame(poses: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Transform global [x, y, heading] poses into the current ego frame."""
    delta = poses[:, :2] - origin[None, :2]
    c, s = np.cos(origin[2]), np.sin(origin[2])
    local_xy = np.stack((c * delta[:, 0] + s * delta[:, 1],
                         -s * delta[:, 0] + c * delta[:, 1]), axis=-1)
    local_heading = wrap_angle(poses[:, 2] - origin[2])
    return np.concatenate((local_xy, local_heading[:, None]), axis=-1)


def rotate_vectors_to_current(vectors: np.ndarray, headings: np.ndarray, current_heading: float) -> np.ndarray:
    """Rotate ego-frame vectors at each timestamp into the current ego frame."""
    vectors = np.asarray(vectors, dtype=np.float64)
    relative = headings - current_heading
    c, s = np.cos(relative), np.sin(relative)
    return np.stack((c * vectors[:, 0] - s * vectors[:, 1],
                     s * vectors[:, 0] + c * vectors[:, 1]), axis=-1)


def interpolate_series(values: np.ndarray, source_times: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    source_times = np.asarray(source_times, dtype=np.float64)
    target_times = np.asarray(target_times, dtype=np.float64)
    if values.ndim == 1:
        return np.interp(target_times, source_times, values)
    return np.stack(
        [np.interp(target_times, source_times, values[:, column]) for column in range(values.shape[1])],
        axis=-1,
    )


def relative_trajectory(rows: list[dict[str, Any]], current_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    global_poses = np.stack([global_pose(row) for row in rows])
    origin = global_poses[current_index]
    local_poses = to_current_frame(global_poses, origin)
    headings = global_poses[:, 2]
    dynamic = np.asarray([row["ego_dynamic_state"] for row in rows], dtype=np.float64)
    velocity = rotate_vectors_to_current(dynamic[:, :2], headings, origin[2])
    acceleration = rotate_vectors_to_current(dynamic[:, 2:4], headings, origin[2])
    return local_poses, velocity, acceleration


def nav_command(driving_command: Any, include_unknown: bool) -> int | None:
    command = np.asarray(driving_command, dtype=np.float32).reshape(-1)
    if command.size < 4:
        return None
    index = int(np.argmax(command[:4]))
    if index == 0:  # NAVSIM [left, straight, right, unknown]
        return 1
    if index == 1:
        return 0
    if index == 2:
        return 2
    return 0 if include_unknown else None


def load_log_index(data_root: Path) -> dict[str, list[dict[str, str]]]:
    index_path = data_root / "indexes" / "sensor_log_index.json"
    if not index_path.exists():
        raise FileNotFoundError(
            f"missing {index_path}; run scripts/index_navsim_shards.py first"
        )
    return json.loads(index_path.read_text())


def image_path(
    data_root: Path,
    log_index: dict[str, list[dict[str, str]]],
    row: dict[str, Any],
    camera: str,
    kind: str,
) -> str | None:
    relative = row.get("cams", {}).get(camera, {}).get("data_path")
    if not relative:
        return None
    for entry in log_index.get(str(row.get("log_name", "")), []):
        if entry["kind"] != kind:
            continue
        shard_root = data_root / Path(entry["root"]).parent
        candidate = shard_root / relative
        if candidate.is_file():
            return candidate.relative_to(data_root).as_posix()
    return None


def row_image_paths(
    data_root: Path,
    log_index: dict[str, list[dict[str, str]]],
    row: dict[str, Any],
    kind: str,
) -> list[str] | None:
    paths = [image_path(data_root, log_index, row, camera, kind) for camera in CAMERAS]
    return paths if all(path is not None for path in paths) else None


def instruction(history: np.ndarray, command: int) -> str:
    labels = ("t-1.5s", "t-1.0s", "t-0.5s", "t-0s")
    lines = "".join(
        f" -{label}: ({pose[0]:.4f}, {pose[1]:.4f}, {pose[2]:.4f});\n"
        for label, pose in zip(labels, history[[0, 5, 10, 15]])
    )
    return (
        "The input images are organized by camera view. Each view contains 4 temporal "
        "frames captured at 0.5s intervals (frame 0 at t-1.5s, frame 3 is the current "
        "frame at t=0s).\n"
        "1. Historical trajectories (x, y, heading) in the current frame's ego coordinate "
        "system. Positive x points forward, positive y points left, and a positive heading "
        "indicates a left turn：\n"
        f"{lines}2. Active navigation command: [{NAV_NAMES[command]}]"
    )


def make_record(
    data_root: Path,
    log_index: dict[str, list[dict[str, str]]],
    window: list[dict[str, Any]],
    command: int,
) -> dict[str, Any] | None:
    current = window[WINDOW_HISTORY - 1]
    history_rows = window[:WINDOW_HISTORY]
    current_paths = row_image_paths(data_root, log_index, current, "current")
    if current_paths is None:
        return None
    history_paths: list[list[str]] = []
    for row in history_rows[:-1]:
        paths = row_image_paths(data_root, log_index, row, "history")
        if paths is None:
            return None
        history_paths.append(paths)

    poses, velocities, accelerations = relative_trajectory(window, WINDOW_HISTORY - 1)
    raw_history_times = np.arange(-1.5, 0.0001, 0.5, dtype=np.float64)
    history_times = np.arange(-1.5, 0.0001, 0.1, dtype=np.float64)
    future_times = np.arange(0.1, 5.0001, 0.1, dtype=np.float64)
    history = interpolate_series(poses[:WINDOW_HISTORY], raw_history_times, history_times)
    history_velocity = interpolate_series(velocities[:WINDOW_HISTORY], raw_history_times, history_times)
    history_acceleration = interpolate_series(accelerations[:WINDOW_HISTORY], raw_history_times, history_times)
    future = interpolate_series(poses[WINDOW_HISTORY - 1 :], np.arange(0.0, 5.0001, 0.5), future_times)

    content: list[dict[str, Any]] = []
    all_paths = [history_paths[0], history_paths[1], history_paths[2], current_paths]
    for label, paths in zip(VIEW_LABELS, zip(*all_paths)):
        for frame_index, path in enumerate(paths):
            content.extend([{"text": label}, {"text": f"frame: {frame_index}"}, {"image": path}])
    content.append({"text": instruction(history, command)})

    dynamic = np.asarray(current["ego_dynamic_state"], dtype=np.float32)
    driving = [int(value) for value in np.asarray(current["driving_command"]).reshape(-1)[:4]]
    return {
        "type": "chatml",
        "messages": [{"role": "user", "content": content}, {"role": "assistant", "content": []}],
        "meta_info": {
            "dataset": "NAVSIM",
            "task_name": "trajectory_prediction",
            "token": str(current["token"]),
            "scene_token": str(current["scene_token"]),
            "log_name": str(current["log_name"]),
            "cam_order": list(CAMERAS),
            "num_history_frames": 4,
            "history_time_offsets": [-1.5, -1.0, -0.5, 0.0],
            "image_hz": 2.0,
            "traj_hz": 10.0,
            "trajectory_source": "openscene_2hz_linear_resample_v1",
        },
        "trajectory": {
            "hist_traj_10hz": history.tolist(),
            "hist_vel_10hz": history_velocity.tolist(),
            "hist_acc_10hz": history_acceleration.tolist(),
            "future_traj_10hz": future.tolist(),
            "future_valid_mask_10hz": [1] * len(future),
            "ego_status": {
                "ego_velocity": dynamic[:2].tolist(),
                "ego_acceleration": dynamic[2:4].tolist(),
                "driving_command": driving,
            },
            "nav_command": command,
        },
        "source": "navtrain",
    }


def iter_records(
    data_root: Path,
    log_index: dict[str, list[dict[str, str]]],
    include_unknown: bool,
    allow_no_route: bool,
    max_pkls: int | None,
    limit: int | None,
) -> tuple[Iterable[dict[str, Any]], dict[str, int]]:
    metadata_dir = data_root / "metadata" / "openscene-v1.1" / "meta_datas" / "trainval"
    paths = sorted(metadata_dir.glob("*.pkl"))
    if max_pkls is not None:
        paths = paths[:max_pkls]
    stats = {"pkls": 0, "windows": 0, "current_missing": 0, "history_missing": 0,
             "unknown": 0, "no_route": 0, "cross_scene": 0, "written": 0}

    def generator() -> Iterable[dict[str, Any]]:
        for pkl_path in paths:
            stats["pkls"] += 1
            with pkl_path.open("rb") as handle:
                rows = pickle.load(handle)
            for start in range(0, len(rows) - WINDOW_SIZE + 1):
                stats["windows"] += 1
                window = rows[start : start + WINDOW_SIZE]
                if len({str(row["scene_token"]) for row in window}) != 1:
                    stats["cross_scene"] += 1
                    continue
                current = window[WINDOW_HISTORY - 1]
                if not allow_no_route and not current.get("roadblock_ids"):
                    stats["no_route"] += 1
                    continue
                command = nav_command(current.get("driving_command"), include_unknown)
                if command is None:
                    stats["unknown"] += 1
                    continue
                record = make_record(data_root, log_index, window, command)
                if record is None:
                    # Distinguish missing current versus history for diagnostics.
                    if row_image_paths(data_root, log_index, current, "current") is None:
                        stats["current_missing"] += 1
                    else:
                        stats["history_missing"] += 1
                    continue
                stats["written"] += 1
                yield record
                if limit is not None and stats["written"] >= limit:
                    return

    return generator(), stats


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    log_index = load_log_index(data_root)
    records, stats = iter_records(
        data_root,
        log_index,
        include_unknown=args.include_unknown,
        allow_no_route=args.allow_no_route,
        max_pkls=args.max_pkls,
        limit=args.limit,
    )
    with output.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    stats_path = output.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "stats": stats}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
