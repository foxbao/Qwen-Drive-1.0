"""Stage 5：解剖一个 Expert 层 —— 它长得像 VLM，而不是像一个回归头

【目的】Qwen-Drive 的专家不是一个「把特征接几层 MLP 出 150 个数」的解码器，
而是一个**照着 VLM 的残差块写的 diffusion-transformer**。这一 stage 把一层拆开看。

真实类是 `planning_expert.PlanningExpertLayer`，一个前向里塞了这些零件：

    ① RMSNorm + **adaLN 调制**（shift / scale）    ← 条件从这里进
    ② 融合 qkv 投影：**每个 query 头自带一个输出门**
    ③ 逐头 RMSNorm（q_norm / k_norm）
    ④ **部分旋转** RoPE（只转前 1/4 个通道）
    ⑤ GQA 注意力 → **门控** → o_proj → 残差
    ⑥ RMSNorm + adaLN 调制 → SwiGLU → 门控 → 残差

【和 Alpamayo 的差异】Alpamayo 的 Expert 是普通的 pre-norm Transformer 块 +
cross-attention 读条件。Qwen-Drive 这一层多了三样东西，全都是从 Qwen3 系 VLM
借过来的：**输出门**、**逐头 qk-norm**、**adaLN-Zero 调制**。原因很实际——
这层要和 VLM 的 KV cache 在同一个注意力和同一套 RoPE 里工作，风格必须对齐。

【adaLN-Zero 是这一层最值得看的设计】调制 MLP 的输出层**零初始化**，
于是训练开始时 6 个调制量全是 0，整层退化成恒等映射：
    x = layernorm(x)*(1+0) + 0 = layernorm(x)
梯度可以从「什么都不做」这个安全的起点慢慢长出来，深层堆叠不会一上来就炸。

【toy 尺寸】hidden=64（真实 1024），heads=4/KV=1（真实 16/4），head_dim=32（真实 256）。
比例和结构一一对应。

【接下来的桥】本章只解释「条件如何被注入」的层内机制；`time + nav + ego_status`
这三个条件向量在哪里构造，要到 Stage 10 才完整展开。先记住：场景内容走联合注意力，
元数据条件走 adaLN 和 waypoint token 两条侧路。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from common import (
    HEAD_DIM,
    HIDDEN,
    INTERMEDIATE,
    NUM_HEADS,
    NUM_KV_HEADS,
    ROTARY_DIM,
    ExpertLayer,
)


def banner(title: str) -> None:
    print("\n" + "=" * 66)
    print(title)
    print("=" * 66)


def main() -> None:
    torch.manual_seed(0)
    layer = ExpertLayer()
    layer.eval()

    batch, length, scene_len = 2, 50, 24
    hidden_states = torch.randn(batch, length, HIDDEN)
    condition = torch.randn(batch, HIDDEN)                       # adaLN 的条件向量
    scene_key = torch.randn(batch, scene_len, NUM_KV_HEADS, HEAD_DIM)
    scene_value = torch.randn(batch, scene_len, NUM_KV_HEADS, HEAD_DIM)
    cos = torch.randn(batch, length, 1, ROTARY_DIM)
    sin = torch.randn(batch, length, 1, ROTARY_DIM)

    banner("① 融合 qkv：一个 Linear 出全部，布局是「VLM 风格」的")
    heads_per_group = NUM_HEADS // NUM_KV_HEADS
    qkv_out = NUM_KV_HEADS * (heads_per_group * 2 + 2) * HEAD_DIM
    print(f"   hidden            {HIDDEN}")
    print(f"   qkv_proj          Linear({HIDDEN} → {qkv_out})")
    print(f"     算法：num_kv_heads × (heads_per_group×2 + 2) × head_dim")
    print(f"           = {NUM_KV_HEADS} × ({heads_per_group}×2 + 2) × {HEAD_DIM} = {qkv_out}")
    print(f"     真实：4 × (4×2 + 2) × 256 = {4 * (4 * 2 + 2) * 256}")
    print(f"\n   注意 (heads_per_group*2 + 2)：**每个 query 头带一个门**，所以 query 部分翻倍。")

    fused = layer.qkv_proj(hidden_states)
    query, gate, key, value = layer.split_qkv(fused)
    print(f"\n   fused             {tuple(fused.shape)}")
    print(f"   拆开后：")
    print(f"     query {tuple(query.shape)}   ← 4 个头 × {HEAD_DIM}")
    print(f"     gate  {tuple(gate.shape)}    ← 和 query 一一对应")
    print(f"     key   {tuple(key.shape)}     ← 只有 {NUM_KV_HEADS} 个头（GQA）")
    print(f"     value {tuple(value.shape)}")
    print(f"\n   每个 KV 组里的排列是 [q0, q1, q2, q3, g0, g1, g2, g3, k, v]，")
    print(f"   所以 `torch.split(..., [per_group*2*head_dim, head_dim, head_dim])` 就能切开，")
    print(f"   再把 query/gate 各 chunk 一刀。省掉 4 次独立矩阵乘。")

    banner("② 输出门：注意力结果先过 sigmoid 再进残差")
    print(f"   真实代码：attn = attn * torch.sigmoid(gate.reshape(...))")
    print(f"   门和 query 头一一对应，是**逐头的标量**（不是逐通道）。")
    print(f"   意义：网络可以选择「这个头这一趟什么都不说」，起到软性抑制的作用。")
    print(f"   sigmoid 而不是 softmax —— 不归一化，允许整体放大或缩小。")

    banner("③ 逐头 qk-norm：在注意力之前把每个头自己归一化")
    print(f"   q_norm / k_norm 都是 RMSNorm({HEAD_DIM})，**只作用在最后一维**。")
    print(f"   真实代码里 q_norm/k_norm 的 weight 是各自独立的参数，不是共享的。")
    print(f"   作用：GQA 下不同头的尺度可能差很多，先归一化再点积，注意力分布更稳。")
    print(f"   副作用：既然做了 qk-norm，RoPE 的作用就只剩「编码相对位置」，")
    print(f"          不再承担「压住数值」的职责 —— 这也是它敢用 rope_theta=1e7 的原因。")

    banner("④ 部分旋转：只转 head_dim 的前 1/4")
    print(f"   head_dim = {HEAD_DIM}，partial_rotary_factor = 0.25")
    print(f"   → rotary_dim = {ROTARY_DIM}，剩下 {HEAD_DIM - ROTARY_DIM} 个通道**原样透传**")
    print(f"\n   真实代码 `_apply_rotary`：")
    print(f"     rotated, passthrough = x[..., :rotary_dim], x[..., rotary_dim:]")
    print(f"     rotated = rotated * cos + rotate_half(rotated) * sin")
    print(f"     return cat([rotated, passthrough], -1)")
    print(f"\n   为什么不全转？位置信息只需要占一部分容量，剩下的留给内容特征。")
    print(f"   这也是 Qwen3 系 VLM 的设定 —— 专家必须跟着走，因为它的 RoPE 参数")
    print(f"   是从 VLM 借来的（stage7 会看到位置也是借的）。")

    banner("⑤ 前向全程的形状追踪")
    modulation = layer.adaln_modulation(condition).chunk(6, dim=-1)
    names = ("shift_attn", "scale_attn", "gate_attn", "shift_ffn", "scale_ffn", "gate_ffn")
    print(f"   condition {tuple(condition.shape)} → 6 个调制向量：")
    for name, tensor in zip(names, modulation):
        print(f"     {name:<12} {tuple(tensor.shape)}")
    print(f"\n   注意调制向量是 **per-sample** 的（batch 维有值、序列维广播），")
    print(f"   和位置无关 —— 它编码的是「这一帧场景整体该怎么去噪」，不是「第几个点」。")

    print(f"\n   {'阶段':<26} {'形状':<26} 说明")
    print(f"   {'-' * 78}")
    x = layer.input_layernorm(hidden_states)
    print(f"   {'input_layernorm':<26} {str(tuple(x.shape)):<26} RMSNorm")
    x = x * (1 + modulation[1].unsqueeze(1)) + modulation[0].unsqueeze(1)
    print(f"   {'adaLN 调制后':<26} {str(tuple(x.shape)):<26} ★(1+scale) 而非 scale")
    q, g, k, v = layer.split_qkv(layer.qkv_proj(x))
    print(f"   {'q / k / v':<26} {str(tuple(q.shape)):<26} GQA: {NUM_HEADS} q 头 : {NUM_KV_HEADS} kv 头")
    joined_k = torch.cat([scene_key, k], dim=1)
    print(f"   {'cat([scene_k, k])':<26} {str(tuple(joined_k.shape)):<26} ★ {scene_len} 场景 + {length} 航点")
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), joined_k.transpose(1, 2), joined_k.transpose(1, 2),
        enable_gqa=True,
    ).transpose(1, 2).reshape(batch, length, -1)
    print(f"   {'注意力输出':<26} {str(tuple(out.shape)):<26} {NUM_HEADS} 头拼回来")
    print(f"   {'o_proj':<26} {str(tuple(layer.o_proj(out).shape)):<26} {NUM_HEADS * HEAD_DIM} → {HIDDEN}")

    banner("⑥ adaLN-Zero：为什么整层初始化成了恒等映射")
    zero_condition = torch.zeros(batch, HIDDEN)
    zeros = layer.adaln_modulation(zero_condition).chunk(6, dim=-1)
    print(f"   调制 MLP 的最后一层权重和偏置都被 zero_ 初始化，所以：")
    print(f"     adaln_modulation(任意输入) = {[round(t.abs().max().item(), 8) for t in zeros]}")
    print(f"\n   代入前向公式：")
    print(f"     x = layernorm(h) * (1 + 0) + 0 = layernorm(h)")
    print(f"     残差 = h + 1 * o_proj(attn)")
    print(f"   —— 一个「只有归一化 + 残差连接」的块，训练从恒等附近出发。")
    print(f"\n   ── 实测：把整层的输出权重也都压成 0，看它是不是真的恒等 ──")
    import copy
    probe = copy.deepcopy(layer)
    probe.eval()
    for module in (probe.o_proj, probe.down_proj):
        torch.nn.init.zeros_(module.weight)
    with torch.no_grad():
        quiet = probe(hidden_states, scene_key, scene_value, cos, sin, zero_condition)
    print(f"   输入与输出最大差 = {(quiet - hidden_states).abs().max().item():.3e}   ← 这就是零初始化的效果")

    banner("⑦ 一次前向的参数量分布（toy vs 真实）")
    print(f"   {'组件':<22} {'toy':>10} {'真实（估算）':>14}")
    print(f"   {'-' * 50}")
    rows = [
        ("qkv_proj", HIDDEN * qkv_out, 1024 * 10240),
        ("o_proj", NUM_HEADS * HEAD_DIM * HIDDEN, 4096 * 1024),
        ("gate_up_proj", HIDDEN * INTERMEDIATE * 2, 1024 * 7168),
        ("down_proj", INTERMEDIATE * HIDDEN, 3584 * 1024),
        ("adaln_modulation", HIDDEN * 6 * HIDDEN, 1024 * 6144),
    ]
    for name, toy, real in rows:
        print(f"   {name:<22} {toy:>10,} {real:>14,}")
    toy_total = sum(r[1] for r in rows)
    real_total = sum(r[2] for r in rows)
    print(f"   {'单层合计':<22} {toy_total:>10,} {real_total:>14,}")
    print(f"   {'× 32 层':<22} {toy_total * 8:>10,} {real_total * 32:>14,}")
    print(f"\n   真实专家约 {real_total * 32 / 1e9:.1f} B 参数，和 docs/model.md「1.0 B」对得上；")
    print(f"   而 planner-rl/model.safetensors 是 2.08 GB → 2.08e9/2 字节 ≈ 1.04 B ✓")
    print(f"\n   注意 adaLN 占了 {real_total and 1024 * 6144 / real_total * 100:.0f}% —— 条件注入不是零成本的小配件。")


if __name__ == "__main__":
    main()
