"""Stage 13：从 50 个点到一个分数 —— 网格转换与指标

【目的】模型**永远**输出同样的一串东西：50 个 (x, y, heading)，5 s @ 10 Hz。
但三个 benchmark 各要各的网格。这一 stage 讲清楚中间的转换，以及指标的定义。

【本 stage 跑的是真代码】`qwen_drive.trajectory` 和 `qwen_drive.metrics` 只依赖 numpy，
不需要 GPU —— 所以可以直接调用，不用 toy 版本。

【输入契约】预测通常是 `[N, 50, 3]`，真值是对应场景文件中的未来轨迹；第 0 个预测点
对应 t=0.1s，不是当前帧。真实数据还可能带 `future_valid_mask`，计算指标前必须按
benchmark 约定处理有效帧；本脚本的合成轨迹没有 mask。Waymo 的 preference/rater、
NAVSIM 的 PDMS 和 `minADE/minFDE` 是不同 selector，不能互相替代。

【三套网格】

    Benchmark      要什么                                怎么转
    ─────────────  ───────────────────────────────────  ──────────────────────────
    NAVSIM v1.1    8 个位姿 @ 2 Hz（t = 0.5 … 4.0 s）   直接取下标 [4, 9, …, 39]
                   （或前 40 个 @ 10 Hz 原样用）
    Waymo E2E      20 个位姿 @ 4 Hz（t = 0.25 … 5.0 s） 插值（heading 走 sin/cos）
    PhysicalAI     50 个 @ 10 Hz                        不转

【一条必须记住的免责声明】`minADE` / `minFDE` 用位移真值挑选，是 oracle 上界。
benchmark best-of-N 的 selector 则因任务而异：Waymo 用 preference/rater 分数，NAVSIM
用 PDMS；两者都不是部署时可直接获得的在线选择信号。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from qwen_drive.metrics import (  # noqa: E402
    navsim_displacement_metrics,
    open_loop_metrics,
    waymo_displacement_metrics,
)
from qwen_drive.trajectory import (  # noqa: E402
    displacement_errors,
    heading_mae,
    resample_10hz_to_2hz,
    resample_to_4hz,
    resample_uniform,
    wrap_angle,
)


def banner(title: str) -> None:
    print("\n" + "=" * 66)
    print(title)
    print("=" * 66)


def make_pair(num: int = 50, noise: float = 0.25, seed: int = 0):
    """造一条预测和一条真值：轻微左弯 + 一点噪声。"""
    rng = np.random.default_rng(seed)
    steps = np.arange(1, num + 1)
    truth = np.stack([steps * 0.7, 0.004 * steps**2, np.arctan2(0.004 * steps**2, steps * 0.7)],
                     axis=-1)
    prediction = truth + rng.normal(0, noise, truth.shape)
    return prediction, truth


def main() -> None:
    prediction, truth = make_pair()

    banner("① 模型的输出永远是 50 个点")
    print(f"   预测形状 {prediction.shape}   真值形状 {truth.shape}")
    print(f"   定义在 `trajectory.py` 的模块文档里：")
    print(f"     「Qwen-Drive predicts 50 waypoints at 10 Hz (5 s).」")
    print(f"   时间戳是 t = 0.1, 0.2, …, 5.0 s —— **从 0.1 开始，不是 0**。")
    print(f"   （第 0 个点是 t=0.1s，所以 50 个点覆盖到 5.0s；真值的第 0 个点才是 t=0 的当前位姿）")

    banner("② NAVSIM：取下标，不插值")
    two_hz = resample_10hz_to_2hz(prediction, horizon_s=4.0)
    print(f"   resample_10hz_to_2hz(pred, horizon_s=4.0) → {two_hz.shape}")
    print(f"   取的是下标 {(np.arange(8) + 1) * 10 // 2 - 1}")
    print(f"\n   公式：index = (k + 1) × source_hz / target_hz − 1，k = 0..7")
    print(f"   这正是 NAVSIM 插值网格的**逆**运算 —— 所以是「取」而不是「插」，")
    print(f"   取出来的点本来就在 NAVSIM 的网格上，没有重采样误差。")

    banner("③ Waymo：插值，而且 heading 要走 sin/cos")
    four_hz = resample_to_4hz(prediction, num_poses=20)
    print(f"   resample_to_4hz(pred, num_poses=20)        → {four_hz.shape}")
    print(f"   目标时间戳 t = 0.25, 0.5, …, 5.0 s，插值出来的。")
    print(f"\n   heading 的插值**不能直接线性插**：")
    a, b = 3.0, -3.0
    direct = (a + b) / 2
    through_sin = np.arctan2((np.sin(a) + np.sin(b)) / 2, (np.cos(a) + np.cos(b)) / 2)
    print(f"     两个朝向 {a} 和 {b} rad（相差 {abs(b - a)} rad ≈ {abs(b-a)*180/np.pi:.0f}°）")
    print(f"     直接线性插值    {direct:+.4f}    ← 完全错了，等于转了大半圈")
    print(f"     走 sin/cos      {through_sin:+.4f}    ← 正确，接近 ±pi")
    print(f"   真实实现 `_interpolate` 里就是这么做的：")
    print(f"     np.interp(sin) 和 np.interp(cos) 分开插，再 arctan2 回来。")

    banner("④ Waymo 的两种约定，结果不一样")
    default = resample_to_4hz(prediction, 20, official_grid=False)
    official = resample_to_4hz(prediction, 20, official_grid=True)
    gap = np.linalg.norm(default[:, :2] - official[:, :2], axis=-1)
    print(f"   default  （归一化下标斜坡）  终点 {np.round(default[-1, :2], 4).tolist()}")
    print(f"   official （绝对时间戳）      终点 {np.round(official[-1, :2], 4).tolist()}")
    print(f"   两者最大位置差 {gap.max():.4f} m   （终点差 {gap[-1]:.6f} m）")
    print(f"\n   default 把两条序列都看成「归一化下标斜坡」，端点严格对齐；")
    print(f"   official 按绝对时间戳插，中间点会更准。")
    print(f"\n   ⚠️ 这里的 {gap.max():.4f} m 是**本 toy 曲线**的重采样 gap，不是 benchmark 固定偏差。")
    print(f"      真实预测应使用目标 benchmark 规定的时间网格重新计算。")
    print(f"      比较 Waymo displacement 数字前，先确认是否使用了 --official-4hz-grid。")
    print(f"\n   差异的来源是「时间轴对齐方式」，不是插值精度：")
    print(f"     default  第 k 个目标点落在原序列的下标 {49 * 2 / 19:.2f}·k 处")
    print(f"     official 第 k 个目标点落在原序列的下标 1.5 + 2.5k 处")
    print(f"   两者只在端点重合（终点差 {gap[-1]:.6f} m 就是这么来的），中间会错开一个下标左右。")

    banner("⑤ 指标：ADE / FDE / minADE —— 以及 min 系列的陷阱")
    distance, ade, fde = displacement_errors(prediction, truth)
    print(f"   displacement_errors 返回 (逐步距离, 平均, 末点)")
    print(f"     ADE = {ade:.4f} m      FDE = {fde:.4f} m")
    print(f"     heading MAE = {heading_mae(prediction, truth):.4f} rad")

    candidates = np.stack([prediction, truth + 0.02, truth + 5.0])   # 三条候选
    print(f"\n   三条候选（第二条好、第三条很离谱）:")
    table = open_loop_metrics(candidates, truth)
    for key in ("ADE_3s", "minADE_3s", "FDE_3s", "minFDE_3s"):
        print(f"     {key:<12} {table[key]:.4f}")
    print(f"\n   minADE/minFDE 只取**最好的那一条**，所以比 ADE/FDE 小得多。")
    print(f"   但「挑最好的」需要知道哪条离真值最近 —— **这要用到真值**。")
    print(f"   所以 min 系列是 **oracle 上界**，不是能部署的数字。")

    banner("⑥ best-of-N 更微妙：selector 随 benchmark 而不同")
    print(f"   两个不同的问题，很容易混：")
    print(f"     minADE  「如果我知道答案，能不能选中最好的那条？」   → 纯上界")
    print(f"     best-of-N  「用 benchmark 自己的打分器选，能得多少分？」 → 依赖打分器")
    print(f"\n   对 Waymo，打分器是 RFS（rater feedback score）——")
    print(f"   它比对**人类评过分的候选轨迹**，所以可以接受一条和实际驾驶不同但同样合理的规划。")
    print(f"   对 NAVSIM，打分器是 PDMS（伪闭环仿真）。")
    print(f"\n   三种选择口径不能混为一种 oracle：")
    print(f"     NAVSIM PDMS  SFT 88.2 → best-of-6 89.3")
    print(f"     Waymo RFS    7.78     → RL 7.91（val 那行是在样本内的）")
    print(f"   `minADE` 用位移真值挑选；Waymo 用 rater preference，NAVSIM 用 PDMS。")
    print(f"   后两者是 benchmark-specific 的选择上界，不等同于部署时可用的在线选择。")

    banner("⑦ 三种网格各自的指标函数")
    print(f"   {'函数':<34} {'用于':<14} {'说明'}")
    print(f"   {'-' * 78}")
    print(f"   {'open_loop_metrics':<34} {'通用':<14} {'多时长 ADE/FDE + min 版本'}")
    print(f"   {'waymo_displacement_metrics':<34} {'Waymo':<14} {'10 Hz + 4 Hz 两套，可选 RFS'}")
    print(f"   {'navsim_displacement_metrics':<34} {'NAVSIM':<14} {'4 s 窗口的 ADE/FDE'}")
    print(f"\n   Waymo 那个函数同时报 10 Hz 和 4 Hz 两组（带 `_10hz` 后缀的是原生的）：")
    waymo = waymo_displacement_metrics(np.stack([prediction]), np.stack([truth]))
    for key in sorted(waymo):
        print(f"     {key:<24} {waymo[key]:.4f}")

    banner("⑧ 哪些指标能离线复现，哪些不能")
    print(f"   {'指标':<24} {'需要什么':<34} {'能否离线'}")
    print(f"   {'-' * 74}")
    print(f"   {'ADE / FDE / minADE':<24} {'预测 + 场景真值（另行读取）':<34} {'✓ 只要 numpy'}")
    print(f"   {'Heading MAE':<24} {'同上，并先处理 valid mask':<34} {'✓'}")
    print(f"   {'Waymo RFS':<24} {'waymo-open-dataset 官方实现':<34} {'✗ 需设置环境变量'}")
    print(f"   {'NAVSIM PDMS':<24} {'navsim + nuplan-devkit + 地图 + 缓存':<34} {'✗ 依赖最重'}")
    print(f"\n   所以 `scripts/eval_physical_ai.py` 是唯一完全自包含的评测脚本。")
    print(f"   `eval_waymo.py` 不加 --rater-feedback 时也只吃 predictions 文件。")
    print(f"   `eval_navsim.py` 没装 navsim 就跑不了 PDMS。")

    banner("⑨ 复现性的三条硬约束（docs 里明写的）")
    print(f"   1. **采样是确定性的**：噪声来自显式播种的 generator，推理阶段贪心解码。")
    print(f"   2. **prompt 是逐字存下来的**：每个 benchmark 场景文件存了当时的 prompt 原文，")
    print(f"      所以模型看到的文本和评测时**完全一致**。")
    print(f"   3. **分辨率会影响轨迹**：每帧的分辨率被记录在场景文件里，先按它缩放再贴 patch 网格。")
    print(f"      换个分辨率 → 视觉 token 变了 → 轨迹就变了。")
    print(f"\n   第 3 条解释了为什么基准分数**不能随便换图源重跑**：")
    print(f"   就算图像内容一样，缩放方式变了结果也会变。")


if __name__ == "__main__":
    main()
