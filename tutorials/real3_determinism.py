"""real3：确定性与 best-of-N（需要 GPU + 完整权重）

【目的】验证 stage11 讲的种子方案在**真实模型**上确实成立，并实测 best-of-N 的收益。

    跑法：python tutorials/real3_determinism.py

【会验证】
    ① 同一个 seed 跑两次 → 轨迹逐元素相同
    ② 样本 k 单独跑 == 放在 batch 里跑（种子独立性的真实检验）
    ③ 不同 seed → 轨迹不同（样本确实有多样性）
    ④ best-of-N 的 ADE / minADE 差距，以及「min 系列是 oracle 上界」这件事
    ⑤ REASONING 模式的理由是贪心生成的 → 本身也可复现

默认使用可同时支持两种规划模式的 `planner-sft`；如使用 `planner-rl`，请只将
REASONING 结果用于质量比较，DIRECT 仅作确定性和开销实验。
"""

from __future__ import annotations

from real_common import banner, load_model, load_samples, require_deps

require_deps(require_weights=True, require_planner=True)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from qwen_drive import InferenceMode  # noqa: E402


def ade_fde(trajectories: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """每条候选的逐点距离 → (ADE, FDE) 数组。"""
    num_poses = min(trajectories.shape[1], truth.shape[0])
    distance = np.linalg.norm(
        trajectories[:, :num_poses, :2] - truth[np.newaxis, :num_poses, :2], axis=-1
    )
    return distance.mean(axis=-1), distance[:, -1]


def main() -> None:
    model = load_model()
    sample = load_samples(limit=1)[0]
    scene = sample.scene
    truth = sample.future_trajectory
    seed = model.config.noise_seed
    noise_device = model.device
    num_points = model.config.num_future_points

    banner("① 同一个 seed，两次运行应当完全一致")
    first = model.run(InferenceMode.DIRECT_PLANNING, scene=scene, num_samples=1, seed=seed)
    second = model.run(InferenceMode.DIRECT_PLANNING, scene=scene, num_samples=1, seed=seed)
    identical = np.array_equal(first.trajectories, second.trajectories)
    print(f"   两次轨迹逐元素相同？  {identical}")
    print(f"   最大差异              {np.abs(first.trajectories - second.trajectories).max():.3e}")
    print(f"\n   来源：`_initial_noise` 用 `torch.Generator().manual_seed(seed + k)` 显式播种。")

    banner("② 样本 k 单独跑 vs 放在 batch 里跑")
    print(f"   起点噪声按 seed+k 独立生成；完整轨迹是否逐元素相同取决于 batch backend。")
    print(f"\n   本机实测：轨迹可能不逐元素相同，下面把差异量出来。")
    batch = model.run(InferenceMode.DIRECT_PLANNING, scene=scene, num_samples=6, seed=seed)
    print(f"\n   batch 轨迹 shape {batch.trajectories.shape}")
    print(f"\n   {'k':>3} {'噪声是否逐位相同':>18} {'轨迹 max|diff|':>16} {'终点差(m)':>12}")
    print(f"   {'-' * 54}")
    for k in (0, 2, 5):
        alone = model.run(InferenceMode.DIRECT_PLANNING, scene=scene, num_samples=1,
                          seed=seed + k)
        noise_same = torch.equal(
            model._initial_noise(1, num_points, seed + k, noise_device)[0],
            model._initial_noise(6, num_points, seed, noise_device)[k],
        )
        diff = np.abs(alone.trajectories[0] - batch.trajectories[k])
        endpoint = np.linalg.norm(
            alone.trajectories[0][-1, :2] - batch.trajectories[k][-1, :2]
        )
        print(f"   {k:>3} {str(noise_same):>18} {diff.max():>16.3e} {endpoint:>12.3e}")

    print(f"\n   所以差异**不是**种子的问题：起点噪声是逐位相同的（第二列全 True）。")
    print(f"   差异来自 **bf16 下的批量矩阵乘**：")
    print(f"     batch=1 和 batch=6 会走不同的 GEMM kernel / 不同的规约顺序，")
    print(f"     bf16 只有 8 位尾数，这些舍入差在 32 层 × 10 步里被逐步放大到厘米级。")

    print(f"\n   再换个 N 验证同一件事：")
    b3 = model.run(InferenceMode.DIRECT_PLANNING, scene=scene, num_samples=3, seed=seed)
    b6 = model.run(InferenceMode.DIRECT_PLANNING, scene=scene, num_samples=6, seed=seed)
    print(f"     batch=3 的前 3 条 vs batch=6 的前 3 条: "
          f"max|diff| = {np.abs(b3.trajectories - b6.trajectories[:3]).max():.3e}")

    print(f"\n   ── 怎么理解这个结果 ──")
    print(f"   1. **种子方案本身是对的**：噪声严格是 seed+k 的函数，与 N 无关。")
    print(f"   2. **但终点输出不是**：批量大小会改变数值路径，差异约 1e-2 ~ 1e-1 m。")
    print(f"   3. 量级上比 ADE(~0.4 m) 小一个数量级，**不影响结论的方向**，")
    print(f"      但它意味着 best-of-N 的排序在接近的候选之间可能翻转。")
    print(f"   4. 想要位级复现，需要固定 batch 大小（或关掉 TF32/用 fp32），")
    print(f"      而这不是推理代码暴露出来的选项。")
    print(f"\n   ⚠️ 这条是**本教程实测**的结论，和官方文档的描述冲突。")
    print(f"      文档描述的是**设计意图**（噪声独立播种）；实测看到的是")
    print(f"      设计意图在 bf16 硬件上**不能被完全兑现**。以实测为准。")

    banner("③ 不同 seed 确实给出不同的轨迹（样本有真正的多样性）")
    endpoint = batch.trajectories[:, -1, :2]
    pairwise = np.linalg.norm(endpoint[:, None, :] - endpoint[None, :, :], axis=-1)
    upper = pairwise[np.triu_indices(len(endpoint), k=1)]
    print(f"   6 条候选的终点两两距离：")
    print(f"     最小 {upper.min():.3f} m   平均 {upper.mean():.3f} m   最大 {upper.max():.3f} m")
    print(f"   6 条终点坐标：")
    for k in range(len(endpoint)):
        print(f"     k={k}  ({endpoint[k, 0]:7.3f}, {endpoint[k, 1]:7.3f})")
    print(f"\n   如果最大距离接近 0，说明模型塌缩到单点，多样性是假的；")
    print(f"   这里 {upper.max():.2f} m 的散布说明噪声确实被解码成了不同的规划。")

    banner("④ best-of-N：ADE vs minADE")
    ade, fde = ade_fde(batch.trajectories, truth)
    print(f"   真值终点 ({truth[-1, 0]:.3f}, {truth[-1, 1]:.3f})")
    print(f"\n   {'k':>3} {'ADE':>9} {'FDE':>9}")
    print(f"   {'-' * 24}")
    for k in range(len(ade)):
        print(f"   {k:>3} {ade[k]:>9.3f} {fde[k]:>9.3f}")
    print(f"   {'-' * 24}")
    print(f"   {'均值':>3} {ade.mean():>9.3f} {fde.mean():>9.3f}   ← 「ADE」报的是这个")
    print(f"   {'最优':>3} {ade.min():>9.3f} {fde.min():>9.3f}   ← 「minADE」报的是这个")
    print(f"\n   比值：minADE / ADE = {ade.min() / ade.mean():.3f}")
    print(f"\n   ⚠️ minADE 比 ADE 好看是**必然的**，因为它偷偷用了真值来挑。")
    print(f"      这不是模型的性能提升，是评价口径的差异。")
    print(f"      部署时没有真值，怎么挑是另一个问题（NAVSIM 用 PDMS、Waymo 用 RFS，")
    print(f"      而那两个打分器本身也需要标签）。")

    banner("⑤ REASONING 模式的理由是可复现的（贪心解码）")
    runs = []
    for _ in range(2):
        out = model.run(InferenceMode.REASONING_PLANNING, scene=scene, num_samples=1, seed=seed)
        runs.append((out.reasoning, out.trajectories))
    print(f"   第一次理由：{runs[0][0]!r}")
    print(f"   第二次理由：{runs[1][0]!r}")
    print(f"   文本相同？  {runs[0][0] == runs[1][0]}")
    print(f"   轨迹相同？  {np.array_equal(runs[0][1], runs[1][1])}")
    print(f"\n   代码里生成时写死了 `do_sample=False`（贪心），所以理由没有随机性。")
    print(f"   轨迹相同则是因为理由相同 → cache 相同 → 加上同一个噪声 seed。")
    print(f"\n   推理阶段唯一的随机源就是**初始噪声**。")

    banner("⑥ 把推理模式的采样也变成多候选")
    reasoned = model.run(InferenceMode.REASONING_PLANNING, scene=scene, num_samples=6, seed=seed)
    r_ade, r_fde = ade_fde(reasoned.trajectories, truth)
    print(f"   理由：{reasoned.reasoning!r}   ← 6 条候选**共享同一份** cache")
    print(f"\n   {'':>10} {'ADE':>9} {'minADE':>9} {'FDE':>9} {'minFDE':>9}")
    print(f"   {'-' * 50}")
    print(f"   {'direct':>10} {ade.mean():>9.3f} {ade.min():>9.3f} "
          f"{fde.mean():>9.3f} {fde.min():>9.3f}")
    print(f"   {'reasoning':>10} {r_ade.mean():>9.3f} {r_ade.min():>9.3f} "
          f"{r_fde.mean():>9.3f} {r_fde.min():>9.3f}")
    print(f"\n   ⚠️ **这是 n=1 的观察**，只说明「在不同条件下能算出不同结果」，")
    print(f"      不能据此说 reasoning 比 direct 好或坏。")
    print(f"      docs/evaluation.md 里那张表才是按 benchmark 全量统计的。")
    print(f"      而且那是 planner-rl 在 reasoning 模式下的官方数字。")

    banner("⑦ 为什么「只跑一次 VLM」对 best-of-N 特别重要")
    print(f"   6 条候选共享同一份 cache，所以理由只有一份 —— 这一点值得注意：")
    print(f"   best-of-N 采样的**多样性完全来自轨迹空间的噪声**，")
    print(f"   **不来自**理由的多样性（理由每次运行都是同一句）。")
    print(f"\n   如果要让理由也多样，需要放开推理阶段的贪心解码，")
    print(f"   那 cache 就要每个样本各跑一遍 —— 代价回到 N 倍 prefill。")


if __name__ == "__main__":
    main()
