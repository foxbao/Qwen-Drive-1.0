"""Stage 3：轨迹表示与归一化 —— 网络内部的「单位」是什么

【目的】专家输出的 50×3 不是米，是**归一化单位**。这一 stage 讲清楚
`trajectory_scale` 怎么来的、为什么要按通道除、以及历史重新参考那一步在干什么。

【和 Alpamayo 的差异】Alpamayo 用单轮车动作空间 (加速度, 曲率)，反归一化是把
mean/std 乘回去再积分。Qwen-Drive **直接回归 (x, y, heading)**，所以归一化就是
逐通道除以一个常数——没有逆运动学，也没有积分误差。

【三个 scale 的来历】
    x: 165 m      —— 未来 5 s 能走多远的上界，留足余量
    y:  25 m      —— 横向位移的上界（换道 / 转弯）
    heading: 1.5703125 rad  —— **这是 pi/2 的 bfloat16 取整值**，不是写错了

第三个最反直觉，也最值得记住：训练时用 bf16 存这个常数，取整误差被固化进了权重，
所以推理必须用同一个数，否则轨迹会系统性偏一点。
"""

from __future__ import annotations

import math

import torch

from common import N_HISTORY, N_HISTORY_QUERY, N_WAYPOINTS, TRAJ_SCALE, TrajectorySpace


def banner(title: str) -> None:
    print("\n" + "=" * 66)
    print(title)
    print("=" * 66)


def sample_trajectory() -> torch.Tensor:
    """造一条形状像真的轨迹：前方 40 m，轻微左偏，heading 慢慢转正。"""
    steps = torch.arange(1, N_WAYPOINTS + 1, dtype=torch.float32)
    x = steps * 0.8
    y = 0.004 * steps**2
    heading = torch.atan2(y, x)
    return torch.stack([x, y, heading], dim=-1).unsqueeze(0)


def main() -> None:
    space = TrajectorySpace()

    banner("① trajectory_scale 的第三个通道：pi/2 的 bf16 取整值")
    pi_over_2 = math.pi / 2
    bf16_rounded = torch.tensor(pi_over_2, dtype=torch.bfloat16).float().item()
    print(f"   math.pi / 2                     = {pi_over_2!r}")
    print(f"   bfloat16(pi/2).float()          = {bf16_rounded!r}")
    print(f"   config 里的 trajectory_scale[2] = {TRAJ_SCALE[2]!r}")
    print(f"   三者相等？{bf16_rounded == TRAJ_SCALE[2]}")
    print(f"\n   bf16 只有 8 位尾数，1.5 ~ 2.0 之间的步长是 2^-7 = {2**-7}，")
    print(f"   pi/2 落到最近的格点上就变成 {TRAJ_SCALE[2]:.7f}。")
    print(f"   训练存的是这个值 → 权重是围着它拟合的 → 推理必须用同一个。")

    banner("② 逐通道归一化：为什么 y 的 scale 比 x 小得多")
    trajectory = sample_trajectory()
    normalized = space.normalize(trajectory)
    print(f"   {'通道':>8} {'原始范围':>22} {'scale':>12} {'归一化后范围':>22}")
    names = ("x", "y", "heading")
    for index, name in enumerate(names):
        raw = trajectory[..., index]
        nrm = normalized[..., index]
        print(f"   {name:>8} [{raw.min():>9.4f}, {raw.max():>9.4f}] "
              f"{TRAJ_SCALE[index]:>12.4f} [{nrm.min():>9.4f}, {nrm.max():>9.4f}]")
    print(f"\n   归一化后所有通道都落在 O(1)：网络不必为「米」和「弧度」用不同的学习率，")
    print(f"   流匹配起点是标准正态 N(0,1)，量纲对齐才能让噪声和数据同一个尺度。")

    banner("③ 往返：normalize → denormalize 应当无损")
    roundtrip = space.denormalize(normalized)
    error = (roundtrip - trajectory).abs().max().item()
    print(f"   最大绝对误差 = {error:.3e}   （float32 精度内无损，heading 会被缠回 [-pi, pi)）")

    banner("④ heading 的缠绕：为什么要 wrap")
    weird = torch.tensor([[[10.0, 0.0, 3.0],
                           [20.0, 1.0, -6.0],
                           [30.0, 2.0, 7.0]]])          # heading 全部越界
    wrapped = space.wrap_heading(weird)
    print(f"   原始 heading      {weird[0, :, 2].tolist()}")
    print(f"   缠绕后            {[round(v, 4) for v in wrapped[0, :, 2].tolist()]}")
    print(f"\n   3.0 rad 和 3.0 - 2*pi = {3.0 - 2 * math.pi:.4f} 是同一个朝向。")
    print(f"   不缠的话，分不清「转了 3 rad」和「转了 -3.28 rad」，回归会把它们当两个目标。")
    print(f"   真实代码在**归一化前和反归一化后各缠一次**：乘法可能把结果再次推出区间。")

    banner("⑤ 历史重参考：不只是归一化，还要换原点")
    # 造一段朝向行驶方向的历史：最早的位姿在 -10 m 处，最新的在原点
    history = torch.zeros(1, N_HISTORY, 3)
    history[0, :, 0] = torch.linspace(-10.0, 0.0, N_HISTORY)
    history[0, :, 1] = torch.linspace(-0.5, 0.0, N_HISTORY)
    print(f"   原始历史 x: [{history[0, :, 0].min():.2f}, {history[0, :, 0].max():.2f}]"
          f"  ← 最老的一帧在 -10 m")
    query = space.normalize_history(history)
    print(f"   normalize_history 输出 {tuple(query.shape)}  (= {N_HISTORY} - 1 = {N_HISTORY_QUERY})")
    print(f"   现在 x 范围: [{query[0, :, 0].min() * TRAJ_SCALE[0]:.2f}, "
          f"{query[0, :, 0].max() * TRAJ_SCALE[0]:.2f}] m  ← 变成往正方向前进")
    print(f"\n   两步操作叠在一起：")
    print(f"     1. 减去最老一帧 → 原点搬到起点，历史变成「朝前走」")
    print(f"     2. 丢掉第 0 行   → 它恒为 (0,0,0)，不含任何信息")
    print(f"   未来本来就是从原点朝前走的，这样**历史和未来的方向约定就统一了**。")

    banner("⑥ 一个容易忽略的细节：历史保持原精度")
    print(f"   DrivingScene.__post_init__ 里 `np.asarray(self.history)` **不带 dtype**，")
    print(f"   是为了一件事：instruction 文本里印的是 4 位小数。")
    print(f"   如果在场景入口就 cast 成 float32，第 4 位小数会和训练数据差一个数字，")
    print(f"   prompt 就变了。专家在真正用到时才 cast 成计算精度——精度只在需要的地方降。")

    banner("⑦ 三条量纲线索汇总")
    print(f"   归一化单位   50×3 张量，量级 O(1)")
    print(f"   物理单位     米 / 弧度，自车当前帧，x 前 y 左 heading 左正")
    print(f"   像素/文本    只出现在 stage1/stage2，专家完全看不见")


if __name__ == "__main__":
    main()
