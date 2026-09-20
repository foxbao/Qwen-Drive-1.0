"""real* 系列共享的工具：加载真实模型 / 真实 demo 数据。

【跑 real* 需要什么】
    GPU（24 GB 够用）+ 真实的 Qwen3.5 权重。
    使用 Python 3.10+、CUDA 和仓库要求的依赖。没有 flash-attn 时使用
    `attn_implementation="sdpa"`；也可以通过命令行参数显式切换 backend。

    `QWEN_DRIVE_MODEL_DIR` 和 `QWEN_DRIVE_PLANNER_DIR` 可用于指定本地权重目录。

【数据】`data/demo/` 自带 4 个 WOD-E2E 场景 + 48 张图。
    注意图在 `data/demo/frames/`，而场景文件里的相对路径是 `frames/scene0_0.jpg`，
    所以 `image_root="data/demo"` 就能直接读 —— **不需要 pyarrow**（用不到 parquet 归档）。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

MODEL_DIR = Path(
    os.environ.get("QWEN_DRIVE_MODEL_DIR", str(ROOT / "models/Qwen-Drive-1.0-4B-ms"))
)
_default_planner = MODEL_DIR / "planner-sft"
if not (_default_planner / "model.safetensors").exists():
    _default_planner = MODEL_DIR / "planner-rl"
PLANNER_DIR = Path(os.environ.get("QWEN_DRIVE_PLANNER_DIR", str(_default_planner)))
SCENES = ROOT / "data/demo/planning_scenes.jsonl"
IMAGE_ROOT = ROOT / "data/demo"

MODEL_LABEL = MODEL_DIR.relative_to(ROOT) if MODEL_DIR.is_relative_to(ROOT) else MODEL_DIR
PLANNER_LABEL = PLANNER_DIR.relative_to(ROOT) if PLANNER_DIR.is_relative_to(ROOT) else PLANNER_DIR
MODEL_README = f"""\
    {MODEL_LABEL}
    ├── model.safetensors          9.1 GB  VLM（共享）
    ├── planner-sft/model.safetensors 2.1 GB 规划专家（DIRECT + REASONING）
    └── planner-rl/model.safetensors  2.1 GB 规划专家（REASONING）

    PLANNER {PLANNER_LABEL}
"""


def require_deps(*, require_weights: bool = False, require_planner: bool = False) -> None:
    """Check dependencies and the files needed by the selected real script."""
    missing = []
    packages = ["torch", "transformers", "PIL"]
    if require_weights:
        packages.extend(("safetensors", "accelerate"))
    for name in packages:
        try:
            __import__(name)
        except ImportError:
            missing.append(name)
    if missing:
        raise SystemExit(
            f"缺少依赖 {missing}。\n"
            f"请用装了 torch / transformers 的解释器运行，例如：\n"
            f"    python {Path(sys.argv[0]).name}"
        )
    for filename in ("config.json", "tokenizer.json"):
        if not (MODEL_DIR / filename).exists():
            raise SystemExit(f"找不到模型文件：{MODEL_DIR / filename}")
    if require_weights and not (MODEL_DIR / "model.safetensors").exists():
        raise SystemExit(
            f"找不到 VLM 权重：{MODEL_DIR / 'model.safetensors'}\n"
            "real1 只检查 config/tokenizer；real2-real4 需要完整 VLM 权重。"
        )
    if require_planner and not (PLANNER_DIR / "model.safetensors").exists():
        raise SystemExit(f"找不到规划专家权重：{PLANNER_DIR / 'model.safetensors'}")


def load_samples(limit: int | None = None):
    """读 demo 场景。"""
    from qwen_drive.benchmarks import read_scene_file

    return list(read_scene_file(SCENES, image_root=IMAGE_ROOT, limit=limit))


def load_model(device: str = "cuda", planner: bool = True):
    """Load the VLM and optional planning head.

    The mixed-mode tutorials prefer planner-sft when it is available; set
    QWEN_DRIVE_PLANNER_DIR to use another expert explicitly.
    """
    import torch

    from qwen_drive import QwenDriveForPlanning

    model = QwenDriveForPlanning.from_pretrained(
        str(MODEL_DIR),
        planner=str(PLANNER_DIR) if planner else None,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    return model.to(device).eval()


def banner(title: str) -> None:
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)
