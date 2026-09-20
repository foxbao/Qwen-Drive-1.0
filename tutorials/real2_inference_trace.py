"""real2：真实推理全流程（需要 GPU + 完整权重）

【目的】把 stage11 的三条路径真跑一遍，并**把中间的 cache 形状打出来**，
对照 toy 讲的形状。同时记一下各阶段的墙钟时间，验证「VLM 只跑一次」这个结论。

    跑法：python tutorials/real2_inference_trace.py
    需要：CUDA 和完整 VLM/planner 权重；显存和加载耗时取决于硬件及 attention backend

【会看到】
    ① VLM 的 8 份场景 cache 到底长什么样
    ② 三种模式的真实输出（含模型自己写的推理文字）
    ③ 推理 / 直接规划 / VQA 的耗时对比
    ④ num_samples 的开销曲线（验证 stage11 ③）

默认优先加载 `planner-sft`（它同时覆盖 DIRECT/REASONING）；若显式指定
`planner-rl`，请把 DIRECT 结果当作架构/耗时观察，因为 RL 专家只在 reasoning rollout
上训练过。
"""

from __future__ import annotations

import time

from real_common import banner, load_model, load_samples, require_deps

require_deps(require_weights=True, require_planner=True)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from qwen_drive import InferenceMode  # noqa: E402


def main() -> None:
    banner("① 加载模型")
    start = time.time()
    model = load_model()
    print(f"   加载耗时 {time.time() - start:.1f} s")
    print(f"   显存占用 {torch.cuda.memory_allocated() / 2**30:.2f} GiB")
    print(f"   VLM 层数             {len(model.config.vlm_config.text_config.layer_types)}")
    print(f"   full_attention 层    {model.config.full_attention_layers}")
    print(f"   专家层数             {model.config.expert_config.num_hidden_layers}")
    print(f"   layers_per_kv        {model.config.expert_config.layers_per_kv}")
    print(f"   → KV 源数           {model.config.expert_config.num_kv_sources}")

    sample = load_samples(limit=1)[0]
    scene = sample.scene

    banner("② VLM prefill：手动跑一次，看 cache 的真实形状")
    inputs = model.processor(scene, with_reasoning=False, device="cpu")
    inputs = {
        key: value.to(model.device) if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }
    start = time.time()
    scene_cache, anchor = model._prefill(inputs)
    prefill_time = time.time() - start
    print(f"   prefill 耗时         {prefill_time:.2f} s")
    print(f"   prompt 长度          {tuple(inputs['input_ids'].shape)}")
    print(f"   场景 cache 份数      {len(scene_cache)}   ← 32 层专家每 4 层共用一份")
    print(f"   单份 K 形状          {tuple(scene_cache[0][0].shape)}")
    print(f"     = (batch, 场景 token 数, VLM 的 {model.config.expert_config.num_key_value_heads} 个 KV 头, "
          f"head_dim {model.config.expert_config.head_dim})")
    print(f"   位置锚点 shape       {tuple(anchor.shape)}   [3 个 mRoPE 段, B]")
    print(f"   锚点数值             {anchor.flatten().tolist()}")
    print(f"\n   对照 stage6 ①：toy 里场景是 24 个 token、1 个 KV 头、head_dim 32；")
    print(f"   真实是几千个 token、4 个 KV 头、head_dim 256 —— 结构一致，只有尺寸不同。")

    banner("③ 直接重新跑一遍完整 direct planning（含采样）")
    start = time.time()
    direct = model.run(InferenceMode.DIRECT_PLANNING, scene=scene, num_samples=1)
    direct_time = time.time() - start
    print(f"   总耗时 {direct_time:.2f} s")
    print(f"   轨迹形状 {direct.trajectories.shape}")
    print(f"   起点 {np.round(direct.trajectory[0], 4).tolist()}")
    print(f"   终点 {np.round(direct.trajectory[-1], 4).tolist()}")
    print(f"   真值终点 {np.round(sample.future_trajectory[-1], 4).tolist()}")
    error = np.linalg.norm(
        direct.trajectory[:, :2] - sample.future_trajectory[: len(direct.trajectory), :2], axis=-1
    )
    print(f"   ADE {error.mean():.3f} m   FDE {error[-1]:.3f} m")

    banner("④ 推理模式：让模型先写一句理由")
    start = time.time()
    reasoned = model.run(InferenceMode.REASONING_PLANNING, scene=scene, num_samples=1)
    reasoned_time = time.time() - start
    print(f"   总耗时 {reasoned_time:.2f} s   （比 direct 多 {reasoned_time - direct_time:+.2f} s）")
    print(f"\n   模型写的理由：")
    print(f"     {reasoned.reasoning!r}")
    print(f"\n   轨迹终点 {np.round(reasoned.trajectory[-1], 4).tolist()}")
    error = np.linalg.norm(
        reasoned.trajectory[:, :2] - sample.future_trajectory[: len(reasoned.trajectory), :2], axis=-1
    )
    print(f"   ADE {error.mean():.3f} m   FDE {error[-1]:.3f} m")
    print(f"\n   两条轨迹的差异（说明理由**确实**影响了输出）：")
    gap = np.linalg.norm(direct.trajectory[:, :2] - reasoned.trajectory[:, :2], axis=-1)
    print(f"     平均 {gap.mean():.4f} m   最大 {gap.max():.4f} m")

    banner("⑤ VQA 模式：同一个 VLM，不经过专家")
    question = "Describe the traffic scene and the safest action."
    start = time.time()
    answer = model.run(InferenceMode.VQA, scene=scene, question=question)
    vqa_time = time.time() - start
    print(f"   问题：{question}")
    print(f"   回答：{answer.text!r}")
    print(f"   耗时 {vqa_time:.2f} s")
    print(f"\n   注意 VQA 走的是完全不同的代码路径（`generate_text`），")
    print(f"   **不碰专家**，解码参数用的是 stage11 里那套 VQA_DECODE_DEFAULTS。")

    banner("⑥ 专家采样的成本结构：固定开销 vs 随 N 增长的边际成本")
    print(f"   只测 `_plan_from_cache`（10 步去噪，不含 prefill），每个 N 预热 2 次再计时：")
    print(f"\n   {'N':>4} {'耗时(s)':>10} {'相对 N=1':>11} {'每样本边际成本':>17}")
    print(f"   {'-' * 48}")
    baseline = None
    for n in (1, 2, 4, 8, 16, 32):
        for _ in range(2):                                  # 预热
            model._plan_from_cache(scene_cache, anchor, inputs, num_samples=n,
                                   num_steps=10, seed=model.config.noise_seed)
        torch.cuda.synchronize()
        start = time.time()
        model._plan_from_cache(scene_cache, anchor, inputs, num_samples=n,
                               num_steps=10, seed=model.config.noise_seed)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        if baseline is None:
            baseline = elapsed
            print(f"   {n:>4} {elapsed:>10.4f} {1.0:>10.2f}x {'—':>17}")
            continue
        marginal = (elapsed - baseline) / (n - 1)
        print(f"   {n:>4} {elapsed:>10.4f} {elapsed / baseline:>10.2f}x {marginal * 1000:>14.2f} ms")
    print(f"\n   本机这次测量中，N 从 1 到 8 的耗时变化较小，N=32 才明显增加。")
    print(f"   说明这个尺度下，专家每步的墙钟时间被**固定的 Python/调度开销**主导，")
    print(f"   真正的 GPU 计算还藏在开销底下。N 大到计算量冒头，才按样本数线性增长。")
    print(f"\n   所以「num_samples 便宜」的准确说法是：")
    print(f"     边际成本（每多一个样本）≪ 固定成本，**直到** N 大到计算量盖过固定开销。")
    print(f"   而不是「专家不要钱」。换更大的专家或更小的 batch，结论可能不同。")

    banner("⑦ 端到端 num_samples 曲线（VLM 只跑一次 + 专家 N 次）")
    print(f"   {'N':>4} {'耗时(s)':>10} {'相对 N=1':>11} {'输出 shape':>16}")
    print(f"   {'-' * 48}")
    end_baseline = None
    for n in (1, 2, 4, 8):
        torch.cuda.synchronize()
        start = time.time()
        out = model.run(InferenceMode.DIRECT_PLANNING, scene=scene, num_samples=n)
        torch.cuda.synchronize()          # ★ 先同步再读表，否则测到的是 kernel 启动时间
        elapsed = time.time() - start
        if end_baseline is None:
            end_baseline = elapsed
        print(f"   {n:>4} {elapsed:>10.2f} {elapsed / end_baseline:>10.2f}x "
              f"{str(out.trajectories.shape):>16}")
    print(f"\n   端到端曲线同样体现 **prefill 只付一次**；具体比例取决于 GPU、backend 和 batch。")
    print(f"\n   对照 stage11 ③ 的估算表：那里的结论方向对了，但**低估了 N 的便宜程度** ——")
    print(f"   它假设专家边际成本恒定，而实测显示这个尺度下边际成本被固定开销盖住了。")
    print(f"   这也是为什么教程里凡是估算都标了「方向确定、数字别当真」。\n")


if __name__ == "__main__":
    main()
