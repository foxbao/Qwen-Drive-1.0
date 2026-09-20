"""Stage 1：场景与坐标系 —— 先把「模型到底看到了什么」钉死

【目的】Qwen-Drive 的全部数字都活在一个坐标系里。这一 stage 把这个坐标系讲清楚，
并把 `DrivingScene` 的每个字段对照 `data/demo/` 里的真实数据看一遍。

【和 Alpamayo 的差异】Alpamayo 的动作是 (加速度, 曲率)，要靠单轮车模型积分成轨迹
（见 alpamayo 的 stage1_action_space）。Qwen-Drive **没有动作空间**：它直接输出
(x, y, heading) 航点。所以这一 stage 的重点从「运动学积分」变成了**坐标系约定与归一化**。

【坐标系（务必记牢）】全部在**当前时刻的自车坐标系**里：
    x  向前为正
    y  **向左**为正        ← 最容易记反
    heading  逆时针为正，左转为正
    单位：米 / 弧度，10 Hz
历史 16 帧（1.5s @ 10Hz），**最后一帧就是当前位姿，恒等于 (0, 0, 0)**。

【本 stage 跑的是真代码】直接 import `qwen_drive.scene`，读 `data/demo/`：
  - 三个相机视角 `<FRONT VIEW>` / `<FRONT LEFT VIEW>` / `<FRONT RIGHT VIEW>`，各 4 个时刻
  - 4 个 demo 场景，覆盖 GO STRAIGHT / TURN LEFT / TURN RIGHT
  - 一个能自证的观察：nav_command 和真值终点方向的符号是对上的
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from qwen_drive.scene import (  # noqa: E402
    CAMERA_VIEWS,
    HISTORY_FRAME_LABELS,
    NAV_COMMANDS,
)


def banner(title: str) -> None:
    print("\n" + "=" * 66)
    print(title)
    print("=" * 66)


def ascii_plot(trajectories: list[tuple[str, np.ndarray]], size: tuple[int, int] = (17, 41)) -> None:
    """把 (x, y) 轨迹画成 ASCII 图：x 向上（前方），y 向右（左侧）。

    注意：屏幕的「上」是 x（前），屏幕的「右」是 y（左）——所以图上往右拐，
    实际是车向左转。这正是 y 轴最容易记反的地方。
    """
    rows, cols = size
    xs = np.concatenate([t[:, 0] for _, t in trajectories])
    ys = np.concatenate([t[:, 1] for _, t in trajectories])
    # 未来只往前走，所以 x 只画 [0, x_max]，把整个高度用满。
    x_max = max(float(xs.max()), 1.0)
    y_max = max(float(np.abs(ys).max()), 1.0)
    grid = [[" "] * cols for _ in range(rows)]
    for index, (_, traj) in enumerate(trajectories):
        for x, y in traj[:, :2]:
            row = int(round((x_max - x) / x_max * (rows - 1)))
            col = int(round((y + y_max) / (2 * y_max) * (cols - 1)))
            grid[row][col] = str(index)
    grid[rows - 1][cols // 2] = "E"          # 自车在当前帧的原点
    print(f"    ↑ 屏幕上方 = x 前方（0 ~ {x_max:.1f} m）")
    print(f"    ↔ 屏幕左右 = y 左侧（±{y_max:.1f} m）；**往右画 = 车往左转**")
    for row in grid:
        print("    |" + "".join(row) + "|")
    print("    " + " " * 5 + "E = 自车当前位置（历史最后一帧 / 未来第 0 帧）")


def main() -> None:
    from qwen_drive.benchmarks import read_scene_file

    root = Path(__file__).resolve().parent.parent
    samples = list(read_scene_file(root / "data/demo/planning_scenes.jsonl",
                                   image_root=root / "data/demo"))

    banner("① 相机排布：3 路 × 4 帧，按「先视角、后时刻」排列")
    print(f"   CAMERA_VIEWS        {CAMERA_VIEWS}")
    print(f"   HISTORY_FRAME_LABELS{HISTORY_FRAME_LABELS}")
    print(f"   顺序 = FRONT 的 4 帧 → FRONT LEFT 的 4 帧 → FRONT RIGHT 的 4 帧")
    print(f"   一共 {len(CAMERA_VIEWS) * 4} 张图。**顺序本身就是信息**：模型靠它区分相机身份。")

    scene = samples[0].scene
    banner("② DrivingScene 字段逐条对照（scene 0）")
    print(f"   history               {scene.history.shape}      16 帧 @10Hz = 1.5 s")
    print(f"   history_velocity      {scene.history_velocity.shape}")
    print(f"   history_acceleration  {scene.history_acceleration.shape}")
    print(f"   ego_velocity          {np.round(scene.ego_velocity, 4).tolist()}  当前帧速度")
    print(f"   ego_acceleration      {np.round(scene.ego_acceleration, 4).tolist()}")
    print(f"   driving_command       {scene.driving_command}   ← 4 维 one-hot")
    print(f"   nav_command           {scene.nav_command}   ← 整数，索引 NAV_COMMANDS")
    print(f"   ego_status            {np.round(scene.ego_status, 3).tolist()}")
    print(f"                         = 速度(2) + 加速度(2) + driving_command(4) = 8 维")

    banner("③ 两个「指令」不是同一个东西（很容易混）")
    print(f"   nav_command     : 整数 0/1/2 → {NAV_COMMANDS}")
    print(f"                     给 **专家** 用：nav_mlp 把它变成 one-hot(3)")
    print(f"   driving_command : 4 维 one-hot，来自 ego_status")
    print(f"                     给 **adaLN 条件** 用：和速度/加速度拼成 ego_status(8)")
    print(f"   两者编码方式、类别数、消费方都不同，demo 里也不总是彼此一致。")

    banner("④ 一个能自证的观察：y 轴向左 —— nav_command 与真值方向对得上")
    print(f"   {'scene':>5}  {'nav_command':>18}  {'真值终点 (x, y)':>18}  解读")
    for index, sample in enumerate(samples):
        scene, gt = sample.scene, sample.future_trajectory
        end = gt[-1]
        name = NAV_COMMANDS[int(scene.nav_command)]
        if name == "GO STRAIGHT":
            reading = f"y={end[1]:+.1f} 基本直行"
        elif name == "TURN LEFT":
            reading = f"y={end[1]:+.1f} **正值 = 向左** ✓"
        else:
            reading = f"y={end[1]:+.1f} **负值 = 向右** ✓"
        print(f"   {index:>5}  {name:>18}  {f'({end[0]:6.2f}, {end[1]:6.2f})':>18}  {reading}")
    print(f"\n   左转的真值终点 y 为正、右转的为负 —— 这条不用信文档，跑一下就能验证。")

    banner("⑤ 把 4 条真值轨迹画出来（屏幕上方 = 前方，屏幕右方 = 左侧）")
    ascii_plot([(f"#{i}", s.future_trajectory[:, :2]) for i, s in enumerate(samples)])

    banner("⑥ 历史重参考：16 帧如何变成专家看到的 15 帧")
    candidates = [
        sample for sample in samples
        if np.linalg.norm(sample.scene.history[:, :2]) > 1e-6
    ]
    if not candidates:
        raise SystemExit("demo 场景中没有历史位姿非零的样本。")
    sample = candidates[0]
    history = sample.scene.history
    print(f"   原始历史 x 范围      [{history[:, 0].min():.3f}, {history[:, 0].max():.3f}]")
    print(f"   最后一帧（当前位姿）  {history[-1].round(4).tolist()}  ← 训练约定里恒为 (0,0,0)")
    print(f"   最老一帧             {history[0].round(4).tolist()}")
    re_referenced = history - history[0:1]
    print(f"   减去最老一帧后       第 0 帧 = {re_referenced[0].round(4).tolist()}，"
          f"它不含信息 → 丢掉")
    print(f"   喂给专家的            {re_referenced[1:].shape}  = 15 个点")
    print(f"   为什么这么做？让历史和未来**都朝行驶方向前进**，网络不必学「历史是倒着的」。")

    banner("⑦ 图像分辨率：历史帧和当前帧不是同一档")
    for label, frame in zip(HISTORY_FRAME_LABELS, scene.views[CAMERA_VIEWS[0]]):
        tag = "当前帧" if label == "t-0s" else "历史帧"
        print(f"   {label:>6} ({tag})  target_size = {frame.target_size}   {frame.image}")
    print(f"\n   当前帧 ~720p、历史帧 ~320p：当前帧的细节更重要，历史帧只需要给运动线索。")
    print(f"   这个 target_size 由 benchmark 元数据给出；没有它才回落到 config 的像素预算")
    print(f"   （history_image_pixels={174080} / current_image_pixels={921600}）。")

    banner("⑧ 指令文本是「合成」出来的，还是「原样读入」的？")
    print(sample.scene.instruction()[:420] + " ...")
    print(f"\n   instruction_text 非空时**原样返回**（benchmark 场景就是这样，保证复现评测时的 prompt）；")
    print(f"   从零构造的场景才会用历史位姿 + nav_command 现拼。两者走的是同一个函数。")


if __name__ == "__main__":
    main()
