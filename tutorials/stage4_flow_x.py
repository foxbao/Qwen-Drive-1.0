"""Stage 4：采样器 —— 干净端点（x）参数化 vs 速度（v）参数化

【目的】这是 Qwen-Drive 和 Alpamayo 差得最「小」、但最值得单独讲的一处。

两边都是 flow matching，都走直线路径，都用欧拉积分。区别只有**网络预测什么**：

    v 参数化（Alpamayo）   网络输出速度 v_hat，更新   x ← x + v_hat * dt
    x 参数化（Qwen-Drive） 网络输出终点 x1_hat，
                          速度现推            v = (x1_hat - x) / (1 - t)
                          更新                x ← x + (x1_hat - x) / (1 - t) * dt

真实代码在 `PlanningExpert.sample`，分母还有个下限 `min_one_minus_t`：

    remaining = max(1 - t, 0.1)
    waypoints = waypoints + (endpoint - waypoints) / remaining * step

【为什么换参数化】
  - 训练目标变成「直接回归终点轨迹」，和最终指标（ADE/FDE 是位置误差）对齐；
  - 在本 stage 的**恒定端点偏差 toy**中，终点误差等于该偏差且不随步数累积；真实网络
    的误差会随状态和时间变化，不能直接套用这个结论；
  - 更新式自动是「朝预测的凸组合」，天然稳定，不需要额外缩放。

【⚠️ 本 stage 只讲采样，不讲训练】这里的 `oracle` 端点函数是**手写的已知答案**，
用来观察积分器的行为；真正可训练的目标见 README §七。
"""

from __future__ import annotations

import torch

from common import MIN_ONE_MINUS_T, N_STEPS, N_WAYPOINTS, TRAJ_SCALE

torch.manual_seed(0)


def banner(title: str) -> None:
    print("\n" + "=" * 66)
    print(title)
    print("=" * 66)


# ---------------------------------------------------------------- 目标轨迹
def target_trajectory(batch: int = 1) -> torch.Tensor:
    """归一化单位下的一条「真值」终点轨迹：走 30 m、轻微左偏。"""
    steps = torch.arange(1, N_WAYPOINTS + 1, dtype=torch.float32)
    x = (steps * 0.6) / TRAJ_SCALE[0]
    y = (0.003 * steps**2) / TRAJ_SCALE[1]
    heading = torch.atan2(y * TRAJ_SCALE[1], x * TRAJ_SCALE[0]) / TRAJ_SCALE[2]
    return torch.stack([x, y, heading], dim=-1).unsqueeze(0).expand(batch, -1, -1).contiguous()


# ---------------------------------------------------------------- 两个积分器
def sample_x(endpoint_fn, noise: torch.Tensor, n_steps: int,
             min_one_minus_t: float = MIN_ONE_MINUS_T) -> torch.Tensor:
    """x 参数化（Qwen-Drive 真实实现）。"""
    x = noise.float()
    step = 1.0 / n_steps
    for index in range(n_steps):
        t = index * step
        endpoint = endpoint_fn(x, t)
        remaining = max(1.0 - t, min_one_minus_t)
        x = x + (endpoint - x) / remaining * step
    return x


def sample_v(velocity_fn, noise: torch.Tensor, n_steps: int) -> torch.Tensor:
    """v 参数化（Alpamayo 那条路），用来对照。"""
    x = noise.float()
    step = 1.0 / n_steps
    for index in range(n_steps):
        # Alpamayo 里 t 是从 1 往 0 走的，这里统一成 0→1，等价。
        x = x + velocity_fn(x, index * step) * step
    return x


def main() -> None:
    x1 = target_trajectory()
    noise = torch.randn_like(x1)

    banner("① 直线路径：x_t = (1-t)·x0 + t·x1，速度恒定")
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        point = (1 - t) * noise + t * x1
        print(f"   t={t:>4}   x_t 第一点 = {point[0, 0].tolist()}"
              f"   （终点处 x1 第一点 = {x1[0, 0].tolist()}）")
    v = x1 - noise
    print(f"\n   整条路径的速度 v = x1 - x0，是个常数场（直线路径的性质）")
    print(f"   v[0, 0] = {v[0, 0].tolist()}")

    banner("② 完美端点预测：默认 10 步下欧拉**精确**落到终点")
    lossless = sample_x(lambda x, t: x1, noise, n_steps=N_STEPS)
    error = (lossless - x1).abs().max().item()
    print(f"   10 步误差 = {error:.3e}")
    print(f"\n   这不是巧合。若端点预测完美，真实解是 x_t = (1-t)x0 + t·x1，")
    print(f"   而欧拉步 x + (x1-x)/(1-t)·dt 恰好等于该解析解在 t+dt 处的取值：")
    print(f"     (x1 - x_t)/(1-t) = (x1 - (1-t)x0 - t·x1)/(1-t) = x1 - x0  ← 正好是常数速度")
    print(f"   所以 x 参数化下，**欧拉的离散误差为 0**，所有误差都来自网络本身。")
    print(f"\n   但这个「零误差」有个前提：分母没有被人为改动。看 ⑤ 那张表就知道，")
    print(f"   n_steps > 10 时 min_one_minus_t 会对最后一步动手：")
    print(f"   {'n_steps':>8} {'最后一步混合系数':>17} {'终点误差':>12}")
    for steps in (4, 6, 10, 12, 20, 50, 100):
        err = (sample_x(lambda x, t: x1, noise, n_steps=steps) - x1).abs().max().item()
        coeff = min(1.0 / steps / MIN_ONE_MINUS_T, 1.0)
        print(f"   {steps:>8} {coeff:>17.3f} {err:>12.3e}")
    print(f"\n   步数 ≤ 10：系数 = 1，精确落点。")
    print(f"   步数 > 10：最后一步只跳 `系数` 这么多，**系统性欠冲**，")
    print(f"             欠冲量 ≈ (1 - 系数) × |x0 - x1|，和起点噪声大小成正比。")
    print(f"   所以「加步数更准」在这里**不成立** —— 多出来的步数被下限挡掉了。")

    banner("③ 对照：v 参数化同样精确（有完美速度场时）")
    v_lossless = sample_v(lambda x, t: v, noise, n_steps=N_STEPS)
    print(f"   10 步误差 = {(v_lossless - x1).abs().max().item():.3e}")
    print(f"   → 完美预测时两者等价。差异只在**预测有误差时怎么传播**，见 ④。")

    banner("④ toy 实验：端点有恒定偏差 ε")
    print(f"   假设网络的终点预测总是偏 ε（一个恒定系统偏差）。")
    print(f"\n   {'ε':>8} {'n_steps':>9} {'终点最大误差':>14} {'误差 - ε':>12} {'是否 = ε':>10}")
    for epsilon in (0.01, 0.05):
        for steps in (10, 20, 50):
            biased = sample_x(lambda x, t, e=epsilon: x1 + e, noise, n_steps=steps)
            err = (biased - x1).abs().max().item()
            print(f"   {epsilon:>8.3f} {steps:>9} {err:>14.5f} {err - epsilon:>12.5f} "
                  f"{str(abs(err - epsilon) < 1e-5):>10}")
    print(f"\n   n_steps = 10：误差**精确等于 ε**。")
    print(f"   推导一下就明白：预测恒为 x1+ε 时，解是 x_t = (1-t)x0 + t·(x1+ε)，")
    print(f"   t=1 处的误差就是 ε —— 和走了几步无关。这是 x 参数化最实用的性质：")
    print(f"   在这个恒定偏差 toy 中，位置指标的误差等于网络端点偏差；真实预测误差不一定恒定。")
    print(f"\n   n_steps = 20 / 50 那几行的 `误差 - ε` 是个正数，来自 ⑤ 说的欠冲，")
    print(f"   不是 ε 被放大了 —— 换个起点噪声这个差值就变，而 ε 那部分始终是 ε。")

    banner("⑤ min_one_minus_t：分母下限什么时候开始咬人")
    print(f"   config: num_inference_steps={N_STEPS}, min_one_minus_t={MIN_ONE_MINUS_T}")
    print(f"\n   {'n_steps':>8} {'step':>8} {'最后一步 t':>11} {'1-t':>8} {'remaining':>10} "
          f"{'混合系数 step/remaining':>22}")
    for steps in (5, 10, 12, 20, 50):
        step = 1.0 / steps
        t_last = (steps - 1) * step
        one_minus_t = 1.0 - t_last
        remaining = max(one_minus_t, MIN_ONE_MINUS_T)
        print(f"   {steps:>8} {step:>8.3f} {t_last:>11.2f} {one_minus_t:>8.3f} {remaining:>10.3f} "
              f"{step / remaining:>22.3f}")
    print(f"\n   混合系数 = 最后一步「有多少比例朝预测端点跳过去」。")
    print(f"     n_steps = 10 → 系数 = 1.0，正好等于 1-t，**下限不生效**（这是默认步数的巧合）")
    print(f"     n_steps > 10 → 系数 < 1.0，最后一步只跳一部分，不再完全信任单次预测")
    print(f"     n_steps < 10 → 系数 = 1.0，但步长变大，积分精度下降")
    print(f"   换句话说：这个下限是为**非默认步数**准备的安全网。")

    banner("⑥ 最后一步预测崩了会怎样：下限的实际作用")
    # 只有当 t 很大（信息最少的那一步反而不对）时才出错的「坏网络」
    def bad_last_step(x, t, bad=0.3):
        if t > 0.9:
            return x1 + bad
        return x1

    for label, floor in (("min_one_minus_t=0.1（真实配置）", MIN_ONE_MINUS_T),
                         ("min_one_minus_t=0.0（去掉下限）", 0.0)):
        result = sample_x(bad_last_step, noise, n_steps=20, min_one_minus_t=floor)
        err = (result - x1).abs().max().item()
        blend = min(1.0 / 20 / floor, 1.0) if floor > 0 else 1.0
        print(f"   {label:<34} 终点误差 {err:.4f}   最后一步混合系数 {blend:.2f}")
    print(f"\n   坏端点偏了 0.3。有下限时混合系数 0.5，只有 0.5×0.3 = 0.15 的坏偏差进得来，")
    print(f"   剩下的是 ⑤ 说的欠冲项；去掉下限则整份 0.3 吃下。")
    print(f"\n   **代价要一起说清楚**：同样的 0.5 系数也作用在好预测上，")
    print(f"   所以默认步数下轨迹会轻微欠冲（端点点不到底）。这是一个稳定性换精度的取舍。")

    banner("⑦ 跑真实实现（common.FlowMatchingX，参数与 config 一致）")
    from common import FlowMatchingX

    sampler = FlowMatchingX()
    print(f"   n_steps={sampler.n_steps}, min_one_minus_t={sampler.min_one_minus_t}")
    result = sampler.sample(lambda x, t: x1, noise)
    print(f"   完美端点 + 真实采样器 → 终点误差 {(result - x1).abs().max().item():.3e}")

    banner("⑧ 起点：噪声是标准正态，不是零")
    print(f"   真实实现 `_initial_noise`：torch.randn(...) * noise_init_std(=1.0)")
    print(f"   每个样本**独立播种**：样本 k 用 seed + k，所以")
    print(f"     - 结果可复现")
    print(f"     - 同一个样本的**起点噪声**单独跑和放在 best-of-N 里逐位相同；完整轨迹可能受 batch 数值路径影响")
    print(f"   起点噪声 shape {tuple(noise.shape)}；num_samples 只改 batch 维，")
    print(f"   VLM 的 scene cache 只算一次然后广播（stage11 详讲）。")


if __name__ == "__main__":
    main()
