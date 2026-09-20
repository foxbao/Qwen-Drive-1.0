"""Stage 11：三种推理模式与 best-of-N —— 钱花在哪儿

【目的】把前面十个 stage 拼成完整的推理路径，并讲清**开销结构**：
为什么说「num_samples 从 1 加到 6，不是 6 倍的代价」。

三种模式（真实枚举 `modeling_qwen_drive.InferenceMode`）：

    VQA                 只用 VLM 回答问题。专家完全不参与，按 Qwen3.5 原样工作。
    DIRECT_PLANNING     user turn 不提推理要求，assistant turn 在 prompt 里直接闭合
                        （stage2 ②），专家从这次 prefill 的 cache 上规划。
    REASONING_PLANNING  VLM 先贪心生成一句理由，把 cache 补到闭合 turn（stage9），
                        专家从**含理由的** cache 上规划。

【开销结构（关键）】
    VLM 前向：一次 prefill 处理几千个 token（其中三千多个是视觉占位符）
    专家前向：50 个 token × 10 步去噪

    num_samples=N 时：
      - VLM 仍然只跑 **1 次**（cache 广播给 N 份，见 `_plan_from_cache` 里的 tile）
      - 专家跑 **N 次**，但每次只有 50 个 token
      → 总耗时通常是固定 prefill 成本 + 专家边际成本；增长方向和比例取决于 backend、GPU 和 batch

【种子方案】`_initial_noise`：样本 k 用 `torch.Generator().manual_seed(seed + k)` 单独播种。
这样起点噪声可复现，且样本 k 单独跑和放在 batch 里跑的**起点噪声**完全一致；
完整轨迹是否逐位一致仍取决于 backend 和 batch kernel（下面 ④ 说明）。
"""

from __future__ import annotations

import torch

from common import N_WAYPOINTS, POINT_DIM

NOISE_SEED = 42
# demo scene 0, DIRECT 的示例计费；真实场景/模式请以运行时 prompt 为准。
IMAGE_TOKENS = 3054
PROMPT_TOKENS = 3385
EXPERT_LAYERS = 32


def banner(title: str) -> None:
    print("\n" + "=" * 66)
    print(title)
    print("=" * 66)


def initial_noise(num_samples: int, num_points: int, seed: int, point_dim: int = POINT_DIM):
    """真实 `_initial_noise` 的复刻：每个样本一个独立的、可复现的 generator。"""
    rows = []
    for offset in range(num_samples):
        generator = torch.Generator().manual_seed(seed + offset)
        rows.append(torch.randn(1, num_points, point_dim, generator=generator))
    return torch.cat(rows, dim=0)


def main() -> None:
    banner("① 三种模式的完整路径")
    print(f"   {'':<12} {'VLM prefill':<14} {'VLM 生成':<18} {'专家':<12} {'输出'}")
    print(f"   {'-' * 72}")
    print(f"   {'VQA':<12} {'1 次':<14} {'可选':<18} {'不用':<12} {'文本'}")
    print(f"   {'DIRECT':<12} {'1 次':<14} {'无':<18} {'10 步 × N':<12} {'轨迹'}")
    print(f"   {'REASONING':<12} {'1 次':<14} {'贪心 ≤256 token':<18} {'10 步 × N':<12} {'轨迹 + 理由'}")
    print(f"\n   注意 DIRECT 和 REASONING 的**专家输入只差 cache 的内容**：")
    print(f"   DIRECT 读的是纯 prompt 的 cache；REASONING 读的是 prompt + 模型自己写的理由。")
    print(f"   同一个专家权重，两种用法。这正是「统一的 VLM + 可插拔的任务头」的体现。")

    banner("② 单次推理的算力账")
    print(f"   一次 DIRECT 规划：")
    print(f"     VLM prefill   {PROMPT_TOKENS:>6} 个 token（含 {IMAGE_TOKENS} 个视觉占位符）")
    print(f"     专家          10 步 × {N_WAYPOINTS} 个 token × {EXPERT_LAYERS} 层")
    print(f"     专家总 token 量 {10 * N_WAYPOINTS * EXPERT_LAYERS:>6} 个（token-层）")
    print(f"\n   VLM 要处理 {PROMPT_TOKENS} 个 token、32 层、hidden 2560；")
    print(f"   专家每步只有 {N_WAYPOINTS} 个 token、32 层、hidden 1024。")
    print(f"   量级上 **专家的一步比 VLM 的 prefill 便宜得多** —— 这是 ③ 的前提。")

    banner("③ num_samples 的代价：VLM 只跑一次")
    print(f"   真实代码 `_plan_from_cache` 里的关键一行：")
    print(f"     def tile(tensor): return tensor.repeat_interleave(num_samples, dim=0)")
    print(f"   场景 cache 只有 1 份，专家侧的所有输入被**复制 N 份**：")
    print()
    print(f"   {'N':>4} {'VLM prefill':>13} {'专家前向':>11} {'相对总开销（粗估）':>20}")
    print(f"   {'-' * 54}")
    # 粗略的相对开销：把 VLM prefill 记 1.0，专家一步记 0.02（数量级示意）
    for n in (1, 2, 4, 6, 12):
        vlm_cost = 1.0
        expert_cost = 0.02 * 10 * n
        print(f"   {n:>4} {1:>13} {10 * n:>11} {vlm_cost + expert_cost:>20.2f}")
    print(f"\n   ⚠️ 上面 0.02 这个系数是**数量级示意**，不是实测。要用 real2 自己测 wall clock。")
    print(f"   结论方向是：N 增大时会复用固定的 VLM prefill，")
    print(f"   但总耗时的具体增长率取决于 GPU、backend、batch 和专家边际成本。")

    banner("④ 种子方案：样本 k 的起点噪声独立于 batch")
    batch = initial_noise(num_samples=6, num_points=N_WAYPOINTS, seed=NOISE_SEED)
    print(f"   6 个样本的噪声 shape {tuple(batch.shape)}")
    print(f"   每个样本来自 seed = {NOISE_SEED} + k：")
    for k in range(4):
        print(f"     k={k}  第一行前 3 个数 {[round(v, 5) for v in batch[k, 0, :3].tolist()]}")
    print(f"     ...")

    alone = initial_noise(num_samples=1, num_points=N_WAYPOINTS, seed=NOISE_SEED + 2)
    identical = torch.equal(alone[0], batch[2])
    print(f"\n   单独取 k=2（seed = {NOISE_SEED + 2}）:  {[round(v, 5) for v in alone[0, 0, :3].tolist()]}")
    print(f"   和 batch 里的第 2 个样本逐元素相同？  {identical}")
    print(f"\n   所以**起点噪声**确实不依赖 N。")

    print(f"\n   ⚠️ 但这里要小心一个常见的过度解读（本教程踩过这个坑）：")
    print(f"      「噪声不依赖 N」**不等于**「输出不依赖 N」。")
    print(f"      完整轨迹是否逐位相同还取决于 batch kernel；real3 会实测这个差异：")
    print(f"      **噪声逐位相同，但轨迹不逐位相同**（差异 ~1e-2 ~ 1e-1 m）。")
    print(f"      原因是 bf16 的批量矩阵乘会随 batch 改变规约顺序，")
    print(f"      舍入差在 32 层 × 10 步里被放大。")
    print(f"      细节和实测数据见 `real3_determinism.py` 的 ②。")
    print(f"\n      量级上这个差异比 ADE 小一个数量级，不影响方向性结论；")
    print(f"      但它意味着**位级复现需要固定 batch 大小**。")

    banner("⑤ 为什么不用「一个 generator 连续抽 6 次」")
    print(f"   一个 generator 连抽 6 次当然也能复现，但你要拿到「样本 2」时，")
    print(f"   必须**先抽掉样本 0 和 1** —— 样本之间有了隐式的顺序依赖。")
    print(f"\n   独立播种没有这个依赖：样本 k 的内容是 seed+k 的纯函数。")
    print(f"   想重放哪一条就重放哪一条，不需要上下文，也不受 N 影响。")
    print(f"\n   对评测很实际：某个场景某条候选特别差，可以直接复现那一条来分析。")

    banner("⑥ 输出的形状与单位")
    print(f"   `QwenDriveOutput`:")
    print(f"     trajectories  np.ndarray  [{N_WAYPOINTS}, 3] 的 (x, y, heading)")
    print(f"                   单位：米 / 弧度，自车当前帧，10 Hz")
    print(f"     reasoning     str 或 None（只有 REASONING 模式有）")
    print(f"     text          str 或 None（只有 VQA 模式有）")
    print(f"\n   `out.trajectory` 是 `trajectories[0]` 的语法糖。")
    print(f"\n   ⚠️ 模型**总是**输出 {N_WAYPOINTS} 个点 × 5 s，不管 benchmark 要什么网格。")
    print(f"      转换在 `qwen_drive.trajectory` 里（stage13 讲）。")

    banner("⑦ 一个容易忽略的点：三种模式共享同一份权重")
    print(f"   `planner-rl` 和 `planner-sft` 是**两个**专家 checkpoint，VLM 是同一个。")
    print(f"   装载方式就是换一个目录：")
    print(f"     model = QwenDriveForPlanning.from_pretrained(vlm, planner='…/planner-rl')")
    print(f"     model.load_planner('…/planner-sft')      # 或者后面再换")
    print(f"\n   docs 里有一句必须注意的：")
    print(f"     「planner-rl was reward-optimized only on reasoning-conditioned rollouts,")
    print(f"       so run it in the reasoning planning mode. planner-sft covers both.」")
    print(f"   即：**RL 专家只在「有理由」的分布上训过**，")
    print(f"   拿它跑 DIRECT 模式属于分布外使用。这是 docs 明说的使用约束。")


if __name__ == "__main__":
    main()
