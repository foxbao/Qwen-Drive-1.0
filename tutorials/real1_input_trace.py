"""real1：真实输入全链路追踪（不需要 GPU 或完整 VLM 权重）

【目的】把 stage1~stage3 用 toy 讲的东西，拿**真数据、真 tokenizer、真 processor**
跑一遍，逐个数字对账。只用 config + tokenizer（权重用 `from_config` 随机初始化，
因为 `get_rope_index` 只依赖 config，不需要真实权重）。

    跑法：python tutorials/real1_input_trace.py

【会验证的四件事】
    ① 12 张图 → 3054 个视觉 token，历史帧 156 / 当前帧 550
    ② DIRECT 与 REASONING 的末尾结构不同，reasoning request 还会增加 token
    ③ mRoPE 锚点是 522，远小于 prompt 长度 3385（stage7 的核心结论）
    ④ 手算锚点 ≈ 实测锚点
"""

from __future__ import annotations

from real_common import SCENES, MODEL_DIR, banner, load_samples, require_deps

require_deps()

import torch  # noqa: E402
from transformers import AutoModelForImageTextToText, AutoTokenizer  # noqa: E402

from qwen_drive.configuration_qwen_drive import QwenDriveConfig  # noqa: E402
from qwen_drive.scene import CAMERA_VIEWS, HISTORY_FRAME_LABELS, QwenDriveProcessor  # noqa: E402


def main() -> None:
    config = QwenDriveConfig.from_pretrained(MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    processor = QwenDriveProcessor(tokenizer, config)
    sample = load_samples(limit=1)[0]
    scene = sample.scene

    banner("① 场景概览")
    print(f"   token                {sample.token}")
    print(f"   相机视角             {len(scene.views)} 路 × {scene.num_camera_frames} 帧")
    print(f"   历史                 {scene.history.shape}  @ {config.trajectory_hz:g} Hz")
    print(f"   nav_command          {scene.nav_command}")
    print(f"   真值未来             {sample.future_trajectory.shape}")
    print(f"   偏好候选             {len(sample.preference_trajectories)} 条, "
          f"分数 {sample.preference_scores}")
    print(f"   初始速度             {sample.initial_speed:.4f} m/s")

    banner("② 图像 → 视觉 token")
    pixel_values, image_grid_thw, token_counts = processor.encode_images(scene)
    print(f"   pixel_values         {tuple(pixel_values.shape)}")
    print(f"   image_grid_thw       {tuple(image_grid_thw.shape)}")
    per_view = scene.num_camera_frames
    print(f"\n   {'#':>3} {'视角':>18} {'帧':>3} {'标签':>7} {'grid(h,w)':>12} {'token':>7}")
    print(f"   {'-' * 58}")
    for index in range(len(token_counts)):
        view = CAMERA_VIEWS[index // per_view]
        frame = index % per_view
        grid = tuple(image_grid_thw[index].tolist()[1:])
        tag = HISTORY_FRAME_LABELS[frame]
        mark = " ←当前" if frame == per_view - 1 else ""
        print(f"   {index:>3} {view:>18} {frame:>3} {tag:>7} {str(grid):>12} "
              f"{token_counts[index]:>7}{mark}")
    total = sum(token_counts)
    print(f"\n   视觉 token 合计      {total}")
    print(f"     算法              grid_h × grid_w / merge_size²  = "
          f"{image_grid_thw[0][1]}×{image_grid_thw[0][2]}/{config.image_spatial_merge_size}² "
          f"= {token_counts[0]}")
    print(f"     merge_size = {config.image_spatial_merge_size}，即 2×2 个 patch 合成 1 个 token")

    banner("③ prompt 组装：两种模式的末尾结构不同")
    for with_reasoning in (False, True):
        ids = processor.build_input_ids(scene, token_counts, with_reasoning)
        mode = "REASONING_PLANNING" if with_reasoning else "DIRECT_PLANNING"
        print(f"\n   {mode}  长度 {len(ids)}")
        print(f"     末尾 60 token: {tokenizer.decode(ids[-60:])!r}")
    direct = processor.build_input_ids(scene, token_counts, False)
    reasoned = processor.build_input_ids(scene, token_counts, True)
    print(f"\n   长度差 = {len(reasoned) - len(direct)} 个 token（那句「请给一句推理」的文本）")

    banner("④ mRoPE 锚点：必须让 VLM 自己算")
    # get_rope_index 只用 config，所以随机初始化的模型足够了
    vlm = AutoModelForImageTextToText.from_config(config.vlm_config)
    modality = (torch.tensor([direct]) == config.vlm_config.image_token_id).long()
    positions, _ = vlm.model.get_rope_index(
        torch.tensor([direct]), mm_token_type_ids=modality, image_grid_thw=image_grid_thw
    )
    anchor = positions[:, :, -1]
    print(f"   rope positions 形状  {tuple(positions.shape)}   [3 段, B, 序列长]")
    print(f"   锚点（最后一个 token）{anchor.tolist()}")
    print(f"   prompt 长度          {len(direct)}")
    print(f"\n   → 锚点 {anchor[0, 0].item():.0f} ≪ prompt 长度 {len(direct)}")
    print(f"     因为 mRoPE 对图像按**网格**编号，不按 token 数编号。")

    print(f"\n   逐段看编号规律（挑几个下标）:")
    image_mask = modality[0] == 1
    print(f"   {'idx':>6} {'是图像':>7} {'t':>6} {'h':>6} {'w':>6}")
    for index in (0, 11, 12, 13, 16, 168, 169, 174, 175, 187, len(direct) - 1):
        print(f"   {index:>6} {str(bool(image_mask[index])):>7} "
              f"{positions[0, 0, index].item():>6.0f} {positions[1, 0, index].item():>6.0f} "
              f"{positions[2, 0, index].item():>6.0f}")

    banner("⑤ 手算锚点，和实测对账")
    n_images = len(token_counts)
    advances = [int(grid[1]) // config.image_spatial_merge_size for grid in image_grid_thw]
    text_tokens = len(direct) - total
    estimate = text_tokens + sum(advances)
    print(f"   纯文本 token                    {text_tokens}")
    print(f"   每张图的推进量（grid_h / merge） {advances}")
    print(f"     合计                         {sum(advances)}")
    print(f"   手算锚点 ≈ {text_tokens} + {sum(advances)} = {estimate}")
    print(f"   实测锚点 = {int(anchor[0, 0].item())}")
    print(f"   差 {abs(estimate - int(anchor[0, 0].item()))}（0-index 的偏移）")
    print(f"\n   所以航点拿到的是位置 {int(anchor[0, 0].item()) + 1} … "
          f"{int(anchor[0, 0].item()) + config.num_future_points}。")

    banner("⑥ 专家侧的输入（绕过 prompt 的另一条路）")
    inputs = processor(scene)
    for key, value in inputs.items():
        print(f"   {key:24s} {tuple(value.shape)}")
    print(f"\n   ⚠️ processor 输出的 history 是 16 帧，**重参考到 15 帧是在专家内部做的**：")
    print(f"      `normalize_history(inputs['history'], scale)` 见 modeling_qwen_drive._plan_from_cache")

    banner("⑦ 归一化常数")
    scale = torch.tensor(config.trajectory_scale)
    print(f"   trajectory_scale     {config.trajectory_scale}")
    print(f"   heading 那一项       {config.trajectory_scale[2]}")
    import math
    print(f"   bfloat16(pi/2)       {torch.tensor(math.pi / 2, dtype=torch.bfloat16).item()}")
    print(f"   相等？               {torch.tensor(math.pi / 2, dtype=torch.bfloat16).item() == config.trajectory_scale[2]}")
    print(f"   num_inference_steps  {config.num_inference_steps}")
    print(f"   min_one_minus_t      {config.min_one_minus_t}")
    print(f"   noise_seed           {config.noise_seed}")


if __name__ == "__main__":
    main()
