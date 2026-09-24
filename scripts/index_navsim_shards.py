#!/usr/bin/env python3
"""Inventory a NAVSIM/OpenScene trainval download without copying sensor files.

The OpenScene download is split into ``navtrain_current_*`` and
``navtrain_history_*`` directories.  This script records where each log lives and,
optionally, checks the three camera references in the trainval metadata.  It writes
small JSON files under ``<data-root>/indexes``; image bytes are never copied.

The metadata files are large, so the default invocation only indexes the sensor
layout.  Use ``--scan-metadata`` for a full metadata availability report, or
``--limit-pkls N`` for a quick smoke check.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


SHARD_RE = re.compile(r"^navtrain_(current|history)_(\d+)$")
CAMERAS = ("CAM_F0", "CAM_L0", "CAM_R0")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="navsim_trainval_v1.1 directory containing sensor_shards/ and metadata/",
    )
    parser.add_argument(
        "--scan-metadata",
        action="store_true",
        help="unpickle trainval metadata and count available target-camera references",
    )
    parser.add_argument(
        "--limit-pkls",
        type=int,
        default=None,
        help="only inspect the first N metadata files (useful for a smoke check)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="where to write index files (default: <data-root>/indexes)",
    )
    return parser.parse_args()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def build_log_index(data_root: Path) -> tuple[dict[str, list[dict[str, str]]], dict[str, Any]]:
    sensor_root = data_root / "sensor_shards"
    if not sensor_root.is_dir():
        raise FileNotFoundError(f"missing sensor shard directory: {sensor_root}")

    log_index: dict[str, list[dict[str, str]]] = {}
    shard_counts: Counter[str] = Counter()
    shard_logs: dict[str, int] = {}
    for shard in sorted(p for p in sensor_root.iterdir() if p.is_dir()):
        match = SHARD_RE.match(shard.name)
        if match is None:
            continue
        kind, number = match.groups()
        logs = [p for p in shard.iterdir() if p.is_dir()]
        shard_counts[kind] += 1
        shard_logs[shard.name] = len(logs)
        for log in logs:
            log_index.setdefault(log.name, []).append(
                {
                    "kind": kind,
                    "shard": shard.name,
                    "shard_number": number,
                    "root": str(log.relative_to(data_root)),
                }
            )

    expected = {"current": 32, "history": 32}
    missing_shards = {
        kind: [str(i) for i in range(1, count + 1)
               if not (sensor_root / f"navtrain_{kind}_{i}").is_dir()]
        for kind, count in expected.items()
    }
    summary = {
        "data_root": str(data_root.resolve()),
        "sensor_root": str(sensor_root.resolve()),
        "num_logs": len(log_index),
        "shards_by_kind": dict(shard_counts),
        "logs_per_shard": shard_logs,
        "missing_expected_shards": missing_shards,
        "logs_with_current_and_history": sum(
            {entry["kind"] for entry in entries} == {"current", "history"}
            for entries in log_index.values()
        ),
    }
    return log_index, summary


def _available(
    data_root: Path,
    entries: list[dict[str, str]],
    relative_path: str,
    kind: str,
) -> bool:
    path = Path(relative_path)
    return any(
        entry["kind"] == kind
        and (data_root / Path(entry["root"]).parent / path).exists()
        for entry in entries
    )


def scan_metadata(
    data_root: Path,
    log_index: dict[str, list[dict[str, str]]],
    output_dir: Path,
    limit_pkls: int | None,
) -> dict[str, Any]:
    metadata_dir = data_root / "metadata" / "openscene-v1.1" / "meta_datas" / "trainval"
    if not metadata_dir.is_dir():
        raise FileNotFoundError(f"missing metadata directory: {metadata_dir}")
    pkls = sorted(metadata_dir.glob("*.pkl"))
    if limit_pkls is not None:
        if limit_pkls < 1:
            raise ValueError("--limit-pkls must be positive")
        pkls = pkls[:limit_pkls]

    manifest_path = output_dir / "metadata_manifest.jsonl"
    totals: Counter[str] = Counter()
    with manifest_path.open("w") as manifest:
        for file_index, pkl_path in enumerate(pkls, start=1):
            with pkl_path.open("rb") as handle:
                rows = pickle.load(handle)
            counts: Counter[str] = Counter()
            missing_logs = 0
            for row in rows:
                counts["rows"] += 1
                entries = log_index.get(str(row.get("log_name", "")), [])
                if not entries:
                    missing_logs += 1
                    continue
                paths = [
                    str(row.get("cams", {}).get(camera, {}).get("data_path", ""))
                    for camera in CAMERAS
                ]
                if not all(paths):
                    counts["missing_camera_metadata"] += 1
                    continue
                current = all(_available(data_root, entries, path, "current") for path in paths)
                history = all(_available(data_root, entries, path, "history") for path in paths)
                mixed = all(
                    any(
                        (data_root / Path(entry["root"]).parent / Path(path)).exists()
                        for entry in entries
                    )
                    for path in paths
                )
                counts["all_current"] += int(current)
                counts["all_history"] += int(history)
                counts["all_any"] += int(mixed)
            counts["missing_log"] = missing_logs
            for key, value in counts.items():
                totals[key] += value
            manifest.write(
                json.dumps(
                    {
                        "metadata": str(pkl_path.relative_to(data_root)),
                        "bytes": pkl_path.stat().st_size,
                        **counts,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            print(
                f"[{file_index}/{len(pkls)}] {pkl_path.name}: "
                f"rows={counts['rows']} current={counts['all_current']} "
                f"history={counts['all_history']} any={counts['all_any']}",
                flush=True,
            )
    return {
        "metadata_dir": str(metadata_dir.resolve()),
        "files_scanned": len(pkls),
        "totals": dict(totals),
        "manifest": str(manifest_path),
    }


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    output_dir = (args.output_dir or data_root / "indexes").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    log_index, summary = build_log_index(data_root)
    _write_json(output_dir / "sensor_log_index.json", log_index)
    report: dict[str, Any] = {"layout": summary}
    if args.scan_metadata:
        report["metadata"] = scan_metadata(
            data_root, log_index, output_dir, args.limit_pkls
        )
    _write_json(output_dir / "layout_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
