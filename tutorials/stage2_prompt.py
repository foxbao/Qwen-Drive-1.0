"""Stage 2：Prompt 是怎么拼出来的 —— 图和话在同一个序列里

【目的】讲清 `QwenDriveProcessor.build_input_ids`：一次规划查询其实就是一个普通的
ChatML 对话，图像不是「另一个输入」，而是**插在文本序列里的定长占位块**。
专家侧的张量（history / ego_status / nav_command）走另一条路，不进 prompt 文本。

【核心结构】一次 DIRECT_PLANNING 的 prompt，从上到下是：

    <|im_start|>user\n
      <FRONT VIEW>            ← 视角标签（纯文本 token）
        frame: 0              ← 时刻标签
        <|vision_start|> <|image_pad|>×156 <|vision_end|>      ← 历史帧，156 个占位
        frame: 1  ...  frame: 2  ...
        frame: 3  <|vision_start|> <|image_pad|>×550 <|vision_end|>   ← 当前帧，550 个占位
      <FRONT LEFT VIEW>   × 4 帧
      <FRONT RIGHT VIEW>  × 4 帧
      <instruction 文本>      ← 历史位姿 + 导航指令（stage1 的 instruction()）
    <|im_end|>\n
    <|im_start|>assistant\n
      [DIRECT 模式]  <|im_end|>\n        ← assistant turn 立刻闭合
      [REASONING 模式] （留空，等模型自己写）← turn 保持打开

**assistant turn 的闭合状态不同**，而 REASONING 还会在 user 文本末尾追加
`REASONING_REQUEST`；两者共同决定 stage9 里 cache 长度和位置锚点的起点。

【为什么占位符数不一样】历史帧和当前帧分辨率不同（stage1 ⑦）：
    历史帧 384×416 → grid 26×24 → 26*24/2² = 156 个 token
    当前帧 799×720 → 贴到 32 的整数倍 → grid 50×44 → 50*44/2² = 550 个 token
    （merge_size=2，所以每 2×2 个 patch 合成 1 个 token）

【本 stage 跑的是真代码】用真 tokenizer + 真 processor，打印真 prompt。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from qwen_drive.configuration_qwen_drive import QwenDriveConfig  # noqa: E402
from qwen_drive.scene import CAMERA_VIEWS, REASONING_REQUEST, QwenDriveProcessor  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models/Qwen-Drive-1.0-4B-ms"

# 这些常量直接决定 prompt 的形状，全部来自 config.json
PATCH_SIZE = 16
MERGE_SIZE = 2
TEMPORAL_PATCH = 2
IMAGE_PAD = "<|image_pad|>"


def banner(title: str) -> None:
    print("\n" + "=" * 66)
    print(title)
    print("=" * 66)


def toy_build_input_ids(processor, scene, token_counts, with_reasoning: bool) -> list[int]:
    """把 `build_input_ids` 的逻辑复写一遍，把「隐藏的字符串拼接」摊开来看。

    和真实实现的唯一区别：这里用文字注释标出每一段在干什么。
    """
    body: list[int] = []
    per_view = scene.num_camera_frames
    for view_index, view in enumerate(CAMERA_VIEWS):
        body += processor._encode(view)                       # "<FRONT VIEW>"
        for frame_index in range(per_view):
            body += processor._encode(f"frame: {frame_index}")  # "frame: 0"
            count = token_counts[view_index * per_view + frame_index]
            body += (
                [processor.vision_start_id]
                + [processor.image_token_id] * count              # 定长占位块
                + [processor.vision_end_id]
            )
    # 指令文本；REASONING 模式在末尾追加一句「请只给一句推理」
    instruction = scene.instruction() + (REASONING_REQUEST if with_reasoning else "")
    body += processor._encode(instruction)

    assistant_header = [processor.im_start_id] + processor._encode("assistant") + processor.newline_ids
    prompt = (
        [processor.im_start_id] + processor._encode("user") + processor.newline_ids
        + body
        + [processor.im_end_id] + processor.newline_ids
        + assistant_header
    )
    if not with_reasoning:
        # DIRECT 模式：assistant turn 立刻闭合，专家从「一个已经结束的空回答」上规划
        prompt += [processor.im_end_id] + processor.newline_ids
    return prompt


def main() -> None:
    from qwen_drive.benchmarks import read_scene_file

    config = QwenDriveConfig.from_pretrained(MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    processor = QwenDriveProcessor(tokenizer, config)
    sample = next(read_scene_file(ROOT / "data/demo/planning_scenes.jsonl",
                                  image_root=ROOT / "data/demo"))
    scene = sample.scene

    banner("① 图像被拍扁成定长占位块")
    pixel_values, image_grid_thw, token_counts = processor.encode_images(scene)
    print(f"   pixel_values        {tuple(pixel_values.shape)}   (所有 patch 拼成一条，无 batch 维)")
    print(f"     单 patch 维度      {pixel_values.shape[1]} = "
          f"3 通道 × {TEMPORAL_PATCH} 时间 × {PATCH_SIZE}×{PATCH_SIZE} patch")
    print(f"   image_grid_thw      {tuple(image_grid_thw.shape)}  每张图的 (t, h, w) 网格")
    print(f"\n   {'顺序':>4} {'视角':>18} {'帧':>4} {'grid (h, w)':>14} {'token 数':>9}")
    per_view = scene.num_camera_frames
    for index, (view, grid, count) in enumerate(
        zip([v for v in CAMERA_VIEWS for _ in range(per_view)], image_grid_thw, token_counts)
    ):
        tag = "当前帧" if index % per_view == per_view - 1 else ""
        print(f"   {index:>4} {view:>18} {index % per_view:>4} "
              f"{str(tuple(grid.tolist()[1:])):>14} {count:>9}  {tag}")
    print(f"\n   总图像 token = {sum(token_counts)}，其中当前帧 "
          f"{sum(token_counts[i] for i in range(per_view - 1, len(token_counts), per_view))}"
          f"、历史帧 {sum(token_counts[i] for i in range(len(token_counts)) if i % per_view != per_view - 1)}")
    print(f"   算一下：grid 50×44 / 2² = {50 * 44 // 4}；grid 26×24 / 2² = {26 * 24 // 4}")

    banner("② 两种模式的 prompt：末尾结构和 reasoning request 都不同")
    for with_reasoning in (False, True):
        ids = processor.build_input_ids(scene, token_counts, with_reasoning)
        mode = "REASONING_PLANNING" if with_reasoning else "DIRECT_PLANNING"
        print(f"\n   --- {mode}  (长度 {len(ids)}) ---")
        print(f"   末尾 90 个 token: {tokenizer.decode(ids[-90:])!r}")
    direct = processor.build_input_ids(scene, token_counts, False)
    reasoned = processor.build_input_ids(scene, token_counts, True)
    print(f"\n   DIRECT    长度 {len(direct)}")
    print(f"   REASONING 长度 {len(reasoned)}   ← 多出的 {len(reasoned) - len(direct)} 个 token 就是那句请求推理的文本")

    banner("③ 自己拼一遍，和真实实现逐 token 对齐")
    mine = toy_build_input_ids(processor, scene, token_counts, with_reasoning=False)
    print(f"   手拼长度 {len(mine)} vs 真实长度 {len(direct)}  →  完全一致：{mine == direct}")
    print(f"\n   前 3 个 token 解出来: {[tokenizer.decode([t]) for t in mine[:3]]}")
    print(f"   vision_start id = {processor.vision_start_id}, "
          f"image_token id = {processor.image_token_id}, im_end id = {processor.im_end_id}")
    print(f"   词表大小 {config.vlm_config.text_config.vocab_size}")
    first_block = mine.index(processor.vision_start_id)
    print(f"\n   第一个 <|vision_start|> 出现在第 {first_block} 个 token，"
          f"前面是 {tokenizer.decode(mine[:first_block])!r}")
    print(f"   → 也就是 '<|im_start|>user\\n<FRONT VIEW>frame: 0'，先文字说清这是哪路相机的第几帧")

    banner("④ 专家侧不走 prompt：另一条并行的输入通道")
    inputs = processor(scene, with_reasoning=False)
    for key, value in inputs.items():
        print(f"   {key:22s} {tuple(value.shape)}")
    print(f"\n   prompt 里的图/文由 VLM 消化；history / ego_status / nav_command "
          f"则**绕过 VLM**，直接进专家（stage10）。")
    print(f"   唯一例外：历史位姿同时也被印成文本放进 instruction —— 双通道，"
          f"文本那份给语言理解，张量那份给轨迹去噪。")

    banner("⑤ 一个反直觉的点：占位符是「定长」的，不是可学习的软 token")
    print(f"   <|image_pad|> 只表示「这里有一个视觉 token」，真实内容由 pixel_values 提供，")
    print(f"   在 VLM 内部被 vision tower 的输出**原地替换**。所以计 token 数 = 数占位符。")
    print(f"   这也解释了为什么改分辨率就改轨迹：占位符数量变了，序列长度和位置都变了。")


if __name__ == "__main__":
    main()
