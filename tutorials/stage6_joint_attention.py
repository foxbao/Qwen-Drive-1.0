"""Stage 6：联合注意力 —— 场景不是「被查询」的，它和航点坐在同一排

【目的】这是 Qwen-Drive 专家最核心的一处设计，一句话：

    attention(Q_waypoints, [K_scene ; K_waypoints], [V_scene ; V_waypoints])

场景 token 和航点 token 的 K/V **拼在一条序列里**，航点在一次注意力里同时
「读场景」和「读彼此」。

【和 Alpamayo 的差异（重点）】

  Alpamayo：cross-attention
      Q = 动作 token（自己算）
      K, V = **只来自** VLM 的条件序列
      → 每个动作 token 独立地从条件里取信息，动作之间在这一层不直接通信

  Qwen-Drive：联合注意力
      Q = 航点 token（自己算）
      K, V = **拼起来**的 [VLM 的场景 cache ; 航点自己的 K/V]
      → 航点既读场景，也读别的航点

【一个直接后果】专家**没有任何 condition 参数**。
不像 cross-attn 需要给 K/V 配一套独立的投影矩阵，这里的场景 K/V 是 VLM 的
「旋转后」产物，直接原地接上，不做任何投影（stage7 讲为什么必须是旋转后的）。

【因果性：causal=False】注意真实代码是 `flash_attn_func(..., causal=False)`。
50 个航点是**一次性一起预测**的，不是一个接一个自回归生成的，
所以航点之间不该有因果 mask —— 第 0 个点看得到第 49 个点，反之亦然。
（对比 Alpamayo 的 CoC 文本生成：那里是自回归的，必须是 causal。）
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from common import HEAD_DIM, HIDDEN, NUM_KV_HEADS, ROTARY_DIM, SCENE_LEN, ExpertLayer, apply_rotary

torch.manual_seed(0)

LENGTH = 50


def banner(title: str) -> None:
    print("\n" + "=" * 66)
    print(title)
    print("=" * 66)


def topologies() -> None:
    print("""
   ┌─ Alpamayo：cross-attention ──────────┐   ┌─ Qwen-Drive：joint attention ────────┐
   │                                      │   │                                      │
   │   Q: 动作 token     (64, d)          │   │   Q: 航点 token      (50, d)         │
   │        │                             │   │        │                             │
   │        ▼                             │   │        ▼                             │
   │   ┌─────────────┐                    │   │   ┌─────────────────────────┐        │
   │   │ 注意力       │◄── K, V 只来自     │   │   │ 注意力                   │◄─ K,V = │
   │   │             │    VLM 条件 (L, d) │   │   │                         │  [场景;  │
   │   └─────────────┘                    │   │   └─────────────────────────┘   航点]  │
   │        │                             │   │        │                  (L+50,d)│
   │        ▼                             │   │        ▼                             │
   │   动作 token 之间在这一层不通信        │   │   航点之间在同一层直接通信             │
   └──────────────────────────────────────┘   └──────────────────────────────────────┘
""")


def main() -> None:
    banner("① 两种拓扑的 K/V 长度对比")
    scene_key = torch.randn(1, SCENE_LEN, NUM_KV_HEADS, HEAD_DIM)
    scene_value = torch.randn(1, SCENE_LEN, NUM_KV_HEADS, HEAD_DIM)
    waypoint_key = torch.randn(1, LENGTH, NUM_KV_HEADS, HEAD_DIM)
    waypoint_value = torch.randn(1, LENGTH, NUM_KV_HEADS, HEAD_DIM)

    cross_kv = scene_key.shape[1]
    joint_kv = scene_key.shape[1] + waypoint_key.shape[1]
    print(f"   scene_key           {tuple(scene_key.shape)}   （真实：几千个 prompt token）")
    print(f"   waypoint_key        {tuple(waypoint_key.shape)}")
    print(f"\n   cross-attn 的 K 长度 = {cross_kv}      （只有场景）")
    print(f"   joint-attn 的 K 长度 = {joint_kv}      （场景 + 航点，多了 50）")
    print(f"\n   代价很小：注意力是 O(L × (L+50))，50 个航点相对几千个场景 token 可以忽略；")
    print(f"   收益是航点之间获得了一次层内的「互相看一眼」。")

    banner("② 关键细节：场景 K/V 不做任何投影")
    print(f"   真实代码 PlanningExpertLayer.forward 里：")
    print(f"     attn = self._attend(query,")
    print(f"                          torch.cat([scene_key, key], dim=1),")
    print(f"                          torch.cat([scene_value, value], dim=1))")
    print(f"\n   `scene_key` / `scene_value` 是**直接传进来的**，没有经过任何 Linear。")
    print(f"   它们的来源是 VLM 分组查询注意力层的 `.keys` / `.values`（旋转之后）。")
    print(f"   所以专家的 `__init__` 里**没有任何处理场景的模块** —— 场景只以 K/V 形式存在。")
    print(f"\n   推论：expert 的 num_key_value_heads / head_dim 必须和 VLM 一模一样，")
    print(f"         否则拼不起来。config 里这两个值不是专家的自由选择，是**继承**的。")

    banner("③ 因果性：50 个航点是一次性一起出来的")
    print(f"   真实代码：flash_attn_func(query, key, value, causal=False)")
    print(f"   另一条实现路径：F.scaled_dot_product_attention(..., enable_gqa=...) 也没传 is_causal。")
    print(f"\n   为什么可以不 causal？")
    print(f"     自回归生成（如 CoC 文本）必须 mask，第 k 个 token 不能看到第 k+1 个。")
    print(f"     但 50 个航点是**同一时刻并行预测**的：噪声 x_T 的 50 行一起进网络，")
    print(f"     50 行预测一起出来。它们没有先后顺序，没有「未来」需要遮住。")
    print(f"\n   如果这里误加了 causal mask 会怎样：第 0 个航点看不到后面 49 个，")
    print(f"   轨迹会被退化成「逐点生成」，而模型从没这样训过 → 输出失配。")

    banner("④ 用真实层做因果探针：扰动一个航点，看别的航点有没有反应")
    print(f"   问题：航点 10 的输出，会不会因为**另一个航点 25 的输入**而改变？")
    print(f"   做法：跑两次真实 ExpertLayer，第二次只把航点 25 的输入加一个常数，")
    print(f"         然后看航点 10 的输出变了多少。")
    print(f"   （RoPE 设成 cos=1 / sin=0，即恒等旋转，排除位置编码的干扰。）")
    print()

    layer = ExpertLayer()
    layer.eval()
    probe_batch, probe_len = 1, LENGTH
    hidden = torch.randn(probe_batch, probe_len, HIDDEN)
    scene_k = torch.randn(probe_batch, SCENE_LEN, NUM_KV_HEADS, HEAD_DIM)
    scene_v = torch.randn(probe_batch, SCENE_LEN, NUM_KV_HEADS, HEAD_DIM)
    cos = torch.ones(probe_batch, probe_len, 1, ROTARY_DIM)
    sin = torch.zeros(probe_batch, probe_len, 1, ROTARY_DIM)
    zero_cond = torch.zeros(probe_batch, HIDDEN)
    source, target = 25, 10

    perturbed = hidden.clone()
    perturbed[0, source] += 1.0
    with torch.no_grad():
        base = layer(hidden, scene_k, scene_v, cos, sin, zero_cond)
        joint_delta = (layer(perturbed, scene_k, scene_v, cos, sin, zero_cond)[0, target]
                       - base[0, target]).abs().max().item()

    def cross_attention_forward(h, sk, sv, c, s, cond):
        """ExpertLayer.forward 的删减版：K/V **只**用场景，不做拼接。"""
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = (
            m.unsqueeze(1) for m in layer.adaln_modulation(cond).chunk(6, dim=-1)
        )
        residual = h
        x = layer.input_layernorm(h) * (1 + scale_a) + shift_a
        q, g, k, v = layer.split_qkv(layer.qkv_proj(x))
        q = apply_rotary(layer.q_norm(q), c, s)
        attn = F.scaled_dot_product_attention(
            q.transpose(1, 2), sk.transpose(1, 2), sv.transpose(1, 2), enable_gqa=True
        ).transpose(1, 2).reshape(h.shape[0], h.shape[1], -1)
        attn = attn * torch.sigmoid(g.reshape(h.shape[0], h.shape[1], -1))
        h = residual + (1 + gate_a) * layer.o_proj(attn)
        residual = h
        x = layer.post_attention_layernorm(h) * (1 + scale_f) + shift_f
        up, gate = layer.gate_up_proj(x).chunk(2, dim=-1)
        return residual + (1 + gate_f) * layer.down_proj(F.silu(up) * gate)

    with torch.no_grad():
        base_x = cross_attention_forward(hidden, scene_k, scene_v, cos, sin, zero_cond)
        cross_delta = (cross_attention_forward(perturbed, scene_k, scene_v, cos, sin, zero_cond)[0, target]
                       - base_x[0, target]).abs().max().item()

    print(f"   {'拓扑':<18} {'航点 10 输出的变化量':>22}")
    print(f"   {'-' * 42}")
    print(f"   {'cross-attention':<18} {cross_delta:>22.6f}")
    print(f"   {'joint attention':<18} {joint_delta:>22.6f}")
    print(f"\n   cross-attn 严格为 0：航点 10 的注意力只读场景，航点 25 改多少都与它无关。")
    print(f"   joint-attn 非 0：航点 25 的 K/V 就在同一条序列里，改它航点 10 立刻有感。")
    print(f"\n   这就是「联合」两个字的全部含义，也是唯一确定的结构性差异 ——")
    print(f"   至于这条通路**被学成了什么**（平滑？互相约束？还是几乎不用？）")
    print(f"   需要看训练后的注意力权重，本教程不做这个断言。")

    banner("⑤ 这条通路为什么对轨迹预测有用")
    print(f"   一条自车轨迹在物理上是**平滑连续**的：相邻 waypoint 的位置/朝向不能突变。")
    print(f"   纯 cross-attn 下，每个航点独立地从场景取信息，只能靠**共享的权重**隐式保证一致性；")
    print(f"   联合注意力让「第 12 个点」直接看到「第 11 和 13 个点」，")
    print(f"   平滑性变成了层内可以显式利用的约束。")
    print(f"\n   另一面：这也意味着**一次前向的 50 个输出不再条件独立**，")
    print(f"   预测出的轨迹更像一条整体采样的曲线，而不是 50 个独立回归结果。")

    banner("⑥ 每个去噪步都要重算：航点 K/V 是 x_t 的函数")
    print(f"   场景 K/V：VLM 前向一次，全程复用（几十个 token 到几千个都一样）")
    print(f"   航点 K/V：**每个去噪步都要重算**，因为 x_t 每步都变")
    print(f"\n   10 步采样 = 10 次完整的 32 层专家前向。这正是 stage11 里")
    print(f"   「num_samples 只增加专家开销、不增加 VLM 开销」这句话的来源。")
    print(f"   VLM 跑 1 次，专家跑 N 次 × 10 步。")


if __name__ == "__main__":
    topologies()
    main()
