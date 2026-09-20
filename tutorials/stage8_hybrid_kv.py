"""Stage 8：混合注意力与 8 份 KV —— 专家为什么只有 32 ← 8 层

【目的】一个很容易看漏、但决定了整个专家拓扑的事实：

    专家的 32 层**不是**一层对一层读 VLM 的 32 层，
    而是每 4 层共用一份 VLM 的 cache，一共只读 **8 份**。

真实代码（两处配合）：

    # configuration_qwen_drive.py
    @property
    def full_attention_layers(self) -> list[int]:
        layer_types = self.vlm_config.text_config.layer_types
        return [i for i, kind in enumerate(layer_types) if kind == "full_attention"]

    # planning_expert.py::predict_endpoint
    scene_key, scene_value = scene_cache[index // self.config.layers_per_kv]

【根因：VLM 是混合注意力的】Qwen3.5-4B 的文本塔有 32 层，但**只有 8 层是真注意力**：

    layer_types = [linear_attention] × 3 + [full_attention] + ... 循环 8 次
                 → 24 层 linear_attention + 8 层 full_attention

`linear_attention`（门控线性注意力 + 短卷积）**不维护逐 token 的 KV cache** ——
它的状态是一个固定大小的递推矩阵，没有「每个 token 一组 K/V」这种东西可读。
所以专家能读的只有那 8 层 full_attention。

【数字怎么对上】
    32 层 VLM ÷ 4（full_attention_interval） = 8 层真注意力
    expert 32 层 ÷ layers_per_kv(=4)         = 8 份 cache
    两边都是 8 → 一一对应。

【和 Alpamayo 的差异】Alpamayo 的 VLM 是普通全注意力堆叠，专家逐层读即可。
这里必须先理解「混合注意力」才知道为什么是 32←8 而不是 32←32。
"""

from __future__ import annotations

import torch

from common import FULL_ATTENTION_LAYERS, LAYERS_PER_KV, NUM_LAYERS, NUM_KV_SOURCES, VLM_LAYERS


def banner(title: str) -> None:
    print("\n" + "=" * 66)
    print(title)
    print("=" * 66)


def main() -> None:
    banner("① VLM 的 32 层里，只有 8 层是真注意力")
    layer_types = [
        "full_attention" if index in FULL_ATTENTION_LAYERS else "linear_attention"
        for index in range(VLM_LAYERS)
    ]
    print(f"   {'层':>4} {'类型':>18}  {'层':>4} {'类型':>18}  {'层':>4} {'类型':>18}  {'层':>4} {'类型':>18}")
    for row in range(8):
        cells = []
        for column in range(4):
            index = row + column * 8
            kind = layer_types[index]
            mark = "★" if kind == "full_attention" else " "
            cells.append(f"{index:>4} {mark}{kind:>17}")
        print("  " + "  ".join(cells))
    full = layer_types.count("full_attention")
    print(f"   合计：linear_attention {layer_types.count('linear_attention')} 层 + "
          f"full_attention {full} 层 = {VLM_LAYERS}")
    print(f"   模式是「3 层线性 + 1 层真注意力」循环，full_attention_interval = 4。")

    banner("② 为什么线性注意力层读不了")
    print(f"   普通注意力（full_attention）每层为每个 token 存一组 K/V：")
    print(f"       缓存大小 ∝ 序列长度          → 可以整段交给别人读")
    print(f"\n   门控线性注意力（linear_attention）把历史压成一个**固定大小的状态矩阵**：")
    print(f"       状态大小 ∝ head_dim²         → 与序列长度无关，但也**没有逐 token 的 K/V**")
    print(f"       它是递推的：S_t = f(S_(t-1), x_t)")
    print(f"\n   专家的联合注意力需要的是「一条可以 cat 上去的 K/V 序列」。")
    print(f"   线性层给不出这种张量，所以只能跳过。这不是取舍，是**类型不匹配**。")

    banner("③ 8 份 cache 怎么分给 32 个专家层")
    print(f"   真实代码用的是整数除法：scene_cache[index // layers_per_kv]")
    print(f"   layers_per_kv = {LAYERS_PER_KV}，所以每 {LAYERS_PER_KV} 个专家层共用一份。")
    print()
    print(f"   {'专家层':>8} {'读到第几份 cache':>18} {'对应 VLM 层':>13}")
    print(f"   {'-' * 46}")
    for expert_layer in range(NUM_LAYERS):
        source = expert_layer // LAYERS_PER_KV
        print(f"   {expert_layer:>8} {source:>18} {FULL_ATTENTION_LAYERS[source]:>13}")
    print(f"\n   推广到真实规模：32 个专家层 → 8 份 cache，"
          f"分别来自 VLM 的第 {FULL_ATTENTION_LAYERS} 层。")
    print(f"   专家层 {NUM_LAYERS} 与 cache 份数 {NUM_KV_SOURCES} 的比值 "
          f"{NUM_LAYERS // NUM_KV_SOURCES} 就是 layers_per_kv。")

    print(f"\n   {'':<24} {'toy':>8} {'真实':>8}")
    print(f"   {'-' * 44}")
    for name, toy, real in (
        ("VLM 层数", VLM_LAYERS, 32),
        ("full_attention 层数", len(FULL_ATTENTION_LAYERS), 8),
        ("专家层数", NUM_LAYERS, 32),
        ("layers_per_kv", LAYERS_PER_KV, 4),
        ("KV 源数", NUM_KV_SOURCES, 8),
    ):
        print(f"   {name:<24} {toy:>8} {real:>8}")

    banner("④ 为什么不干脆让 32 个专家层各配一份")
    print(f"   可以，但没东西可配 —— VLM 只有 8 层真注意力，一共就 8 份 cache。")
    print(f"   反过来，也可以让专家层数 = cache 份数（8 层专家），但那样太浅了：")
    print(f"     专家的容量主要花在**去噪**上，需要足够的深度做 10 步迭代；")
    print(f"     而每一层读同一份场景 cache 是**完全合理**的 —— 场景没变，")
    print(f"     变的是航点自身，让不同深度的层反复以不同方式读同一份场景，")
    print(f"     和 VLM 里「同一份输入过很多层」是一个道理。")
    print(f"\n   换句话说：layers_per_kv 这个参数**不是**为了省显存，")
    print(f"   它只是把「专家要多深」和「VLM 提供几份 cache」这两件事解耦了。")

    banner("⑤ 一个真实约束：改不了")
    print(f"   config 注释：「The expert reads the post-rotary keys/values of the VLM's")
    print(f"   grouped-query attention layers. kv_head_dim and num_key_value_heads")
    print(f"   therefore have to match the VLM exactly.」")
    print(f"\n   连带不能改的还有：")
    print(f"     num_key_value_heads = 4   （VLM 的 KV 头数，拼 K/V 时维度要对齐）")
    print(f"     head_dim            = 256 （同上）")
    print(f"     mrope_section / rope_theta / partial_rotary_factor  （stage7）")
    print(f"     layers_per_kv       = 4   （和 full_attention_interval 对齐）")
    print(f"\n   可以自由改的：hidden_size（1024）、中间层宽度（3584）、专家层数（32）。")
    print(f"   这解释了为什么 docs/model.md 把这些参数单列成一张「dictated by the VLM」的表。")

    banner("⑥ 顺手确认一下：8 份 cache 的形状确实和专家对得上")
    scene_len = 24
    cache = [(torch.randn(1, scene_len, 4, 256), torch.randn(1, scene_len, 4, 256))
             for _ in range(8)]
    print(f"   scene_cache 长度      {len(cache)}")
    print(f"   每份 K 的形状          {tuple(cache[0][0].shape)}")
    print(f"     = (batch, 场景 token 数, VLM 的 KV 头数 4, head_dim 256)")
    print(f"\n   专家侧 waypoint K 的形状是 (batch, 50, 4, 256) —— 后两维完全一致，")
    print(f"   所以 `torch.cat([scene_key, key], dim=1)` 才能在**序列维**上拼接。")
    print(f"   真实场景里第 2 维是几千个 token，而不是这里的 {scene_len}。")


if __name__ == "__main__":
    main()
