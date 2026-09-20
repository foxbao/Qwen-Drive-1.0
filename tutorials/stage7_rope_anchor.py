"""Stage 7：位置锚点 —— 航点 token 站在 VLM 的「下一格」上

【目的】50 个航点 token 需要位置编码。Qwen-Drive 的做法不是学一个位置表，而是
**接着 VLM 的序列往下排**：第 k 个航点的位置 = 锚点 + k + 1。

本章的规则对所有场景都成立；下面的 3385、3054、522 只是 demo scene 0 的
DIRECT 实测值。真实代码始终从当前 prompt 动态计算 anchor，不能把 522 当模型常量。

真实代码只有三行：

    # modeling_qwen_drive.py::_prefill
    anchor = self._rope_positions(input_ids, inputs["image_grid_thw"])[:, :, -1]
    # → 形状 [3, B]：三个 mRoPE 段，各取 prompt 最后一个 token 的位置

    # planning_expert.py::_waypoint_positions
    steps = torch.arange(1, length + 1)
    return anchor.unsqueeze(-1) + steps        # → [3, B, 50]

【和 Alpamayo 的差异】Alpamayo 的 Expert 用**学习到的**位置/索引编码（stage3 的
Fourier 编码 + 可学习 embedding）。Qwen-Drive 用**继承来的** RoPE 位置：
航点 token 在旋转相位上就是「prompt 后面的第 1..50 个 token」。

【为什么必须这样做】专家的注意力里，场景 K/V 是 VLM **旋转之后**的产物
（`layer.keys`，post-rotary）。RoPE 的旋转是相对的：只有当航点自己的 RoPE 相位
和场景 token 的相位在**同一个坐标系**里，内积才能正确编码相对距离。
另起一套位置编码会让 query 和 key 的相位对不上，注意力退化。

【mRoPE 是什么】Qwen3.5 是多模态模型，位置是**三维**的 (t, h, w)，
对应三个 section (11, 11, 10)，交错地填进 32 个频率对（rotary_dim 64 / 2）。
本 stage 用 toy 的 (2, 1, 1) 把这套交错规则演示清楚。
"""

from __future__ import annotations

import torch

from common import MROPE_SECTION, ROTARY_DIM, WaypointRotaryEmbedding

LENGTH = 50
# 下面四个数是在 data/demo 的 scene 0 上**实测**出来的（DIRECT 模式）。
PROMPT_LEN = 3385          # prompt token 数
IMAGE_TOKENS = 3054        # 其中视觉占位符
REAL_ANCHOR = 522          # get_rope_index 给出的最后一个 token 位置


def banner(title: str) -> None:
    print("\n" + "=" * 66)
    print(title)
    print("=" * 66)


def main() -> None:
    banner("① 锚点从哪来：不是 prompt 长度，mRoPE 的位置空间紧凑得多")
    print(f"   `get_rope_index` 返回 [3, B, S]：三个 mRoPE 段 × batch × 序列长度")
    print(f"   取 [:, :, -1] → [3, B]，就是把**最后一个 token** 的位置摘出来。")
    print(f"\n   实测（demo scene 0, DIRECT 模式）：")
    print(f"     prompt token 数                    {PROMPT_LEN}")
    print(f"     其中视觉占位符                     {IMAGE_TOKENS}")
    print(f"     纯文本 token                       {PROMPT_LEN - IMAGE_TOKENS}")
    print(f"     **最后一个 token 的 mRoPE 位置**    {REAL_ANCHOR}")
    print(f"\n   锚点只有 {REAL_ANCHOR}，不是 {PROMPT_LEN - 1}。差了 {PROMPT_LEN // REAL_ANCHOR} 倍多。")
    print(f"   所以位置**绝对不能**用「第几个 token」去猜，必须让 VLM 自己算：")
    print(f"     positions, _ = self.vlm.model.get_rope_index(input_ids, ...)")
    print(f"     anchor = positions[:, :, -1]")

    banner("①b 为什么位置空间这么紧凑：图像按「格」计费，不按 token 计费")
    print(f"   看真实的 mRoPE 编号轨迹（demo scene 0，前两块）：")
    trace = [
        (0, 0, 0, 0, 0, "文本：t=h=w 一起 +1"),
        (11, 0, 11, 11, 11, "文本"),
        (12, 0, 12, 12, 12, "文本（最后一个文本）"),
        (13, 1, 13, 13, 13, "★ 图像块开始：t 冻住"),
        (16, 1, 13, 13, 16, "  w 沿着网格推进"),
        (168, 1, 13, 25, 24, "  图像块结束（13×12 个 token）"),
        (169, 0, 26, 26, 26, "★ 下一个文本：max(h,w)+1 = 26"),
        (174, 0, 31, 31, 31, "文本"),
        (175, 1, 32, 32, 32, "★ 下一张图，从 32 开始"),
        (186, 1, 32, 32, 43, "  w 走到 43 就回卷"),
        (187, 1, 32, 33, 32, "  h 进一格，w 回到 32"),
    ]
    print(f"   {'idx':>6} {'img':>4} {'t':>5} {'h':>5} {'w':>5}  说明")
    for idx, img, t, h, w, note in trace:
        print(f"   {idx:>6} {img:>4} {t:>5} {h:>5} {w:>5}  {note}")
    print(f"\n   规律：")
    print(f"     文本 token：t == h == w，每个 token 三个段一起 +1")
    print(f"     图像 token：t **冻住不变**，(h, w) 在二维网格上推进")
    print(f"\n   一张 patch 网格 (26, 24) 的历史帧有 {26 * 24 // 4} 个 token，")
    print(f"   但它的位置网格只有 (26/2, 24/2) = (13, 12) —— 这正是 merge_size=2 的效果：")
    print(f"   **2×2 个 patch 合成 1 个 token**，位置也按合并后的格子计。")
    print(f"   于是这张图在经济上只推进了 13 格（h 走 13 行），而不是 156 格。")

    # 用手算验证锚点
    history_advance = 13      # 历史帧 token 网格 13×12 → 推进 13
    current_advance = 25      # 当前帧 token 网格 25×22 → 推进 25
    text_tokens = PROMPT_LEN - IMAGE_TOKENS
    estimate = text_tokens + 9 * history_advance + 3 * current_advance
    print(f"\n   手算一遍锚点：")
    print(f"     纯文本            {text_tokens}")
    print(f"     9 张历史帧 × {history_advance}    {9 * history_advance}")
    print(f"     3 张当前帧 × {current_advance}    {3 * current_advance}")
    print(f"     合计 ≈ {estimate}   （实测 {REAL_ANCHOR}，差 1 是 0-index 的偏移）")
    print(f"\n   所以航点拿到的是位置 {REAL_ANCHOR + 1} … {REAL_ANCHOR + LENGTH}，")
    print(f"   在**紧凑的位置空间**里紧挨着 prompt，而不是在 {PROMPT_LEN} 后面。")

    anchor = torch.tensor([[float(REAL_ANCHOR)], [float(REAL_ANCHOR)], [float(REAL_ANCHOR)]])

    banner("② 航点位置 = 锚点 + 1 … 锚点 + 50")
    steps = torch.arange(1, LENGTH + 1, dtype=anchor.dtype)
    positions = anchor.unsqueeze(-1) + steps
    print(f"   steps      {steps[:5].tolist()} ... {steps[-3:].tolist()}")
    print(f"   positions  {tuple(positions.shape)} = [3 段, B, {LENGTH}]")
    for section in range(3):
        row = positions[section, 0]
        print(f"     段 {section}: 第 0 个航点 @ {row[0].item():.0f}, "
              f"第 49 个 @ {row[-1].item():.0f}")

    banner("③ 「最后一个是文本 token，所以三段同锚点」")
    print(f"   mRoPE 对图像 token 会按 (t, h, w) 分别编号，三段的值**不相等**；")
    print(f"   而文本 token 的 t/h/w 是一起递增的，三段**相等**。")
    print(f"\n   真实代码注释里写得很直白：")
    print(f"     「Waypoint tokens are positioned immediately after the VLM prefix, so")
    print(f"       their rotary phases continue the language model's. All three mRoPE")
    print(f"       sections share the same anchor because the last prefix token is")
    print(f"       always a text token.」")
    print(f"\n   代码上表现为 `torch.arange(1, length+1)` 是**一维**的：")
    print(f"   三个段加的是同一个 steps，而不是像图像那样各自走网格。")
    print(f"\n   另外一个细节：attention 结束时 prompt 的最后一段是")
    print(f"   '<|im_start|>assistant\\n' —— 确实是纯文本，前提成立。")

    banner("④ mRoPE 的交错填充规则")
    emb = WaypointRotaryEmbedding()
    print(f"   head_dim = 32, partial_rotary_factor = 0.25 → rotary_dim = {ROTARY_DIM}")
    print(f"   频率对数 = rotary_dim / 2 = {ROTARY_DIM // 2}")
    print(f"   mrope_section = {tuple(MROPE_SECTION)}，和正好 = {sum(MROPE_SECTION)} ✓")
    print(f"   （真实是 (11, 11, 10)，和 = 32 = 256×0.25/2）")
    print()
    print(f"   填充规则（真实代码 `merged[..., offset : length*3 : 3] = angles[offset][...]`）：")
    total = ROTARY_DIM // 2
    # Match the production slice assignments rather than assuming three equal sections.
    owner = [0] * total
    for section, length in enumerate(MROPE_SECTION[1:], start=1):
        for index in range(section, length * 3, 3):
            owner[index] = section
    print(f"     频率对下标:  {list(range(total))}")
    print(f"     来自哪一段:  {owner}")
    print(f"\n   这组 toy section 下，段 0 管 0, 3, ...；段 1 管 1；段 2 管 2。")
    print(f"   真实 `(11, 11, 10)` 则会交错覆盖更多频率对。**注意不是分块**：每段的长度")
    print(f"   决定它在交错序列中占多少位置。")

    banner("⑤ 实际算一遍：航点 0 的 cos/sin")
    cos, sin = emb(positions[:, :1, :2], torch.float32)
    print(f"   输入 positions 形状 {tuple(positions[:, :1, :2].shape)} → [3, B, L]")
    print(f"   输出 cos {tuple(cos.shape)}, sin {tuple(sin.shape)}   [B, L, 1, rotary_dim]")
    print(f"\n   注意中间那个 **1**：它是给注意力头留的广播维。")
    print(f"   RoPE 是**每个头共享同一份相位**的（位置和头无关），所以只占 1 维。")
    print(f"   对照 q 的形状 [B, L, heads, head_dim] —— 广播上去正好对齐。")

    banner("⑥ 旋转是相对的：相位差才有意义")
    print(f"   RoPE 通过 q·k 的旋转内积编码**相对**位置：")
    print(f"     位置 p 的 q 与位置 q 的 k 内积 ∝ f(p - q)")
    print(f"   所以航点 0 和航点 49 之间的关系，只取决于它们相差 49，")
    print(f"   而**不是**取决于锚点是 3384 还是 5000。")
    print()
    print(f"   这说明航点之间的相对结构在公式上不随锚点平移而改变；把它解释为模型的")
    print(f"   外推能力仍是推论，需要实测验证。")
    print(f"\n   ⚠️ 但**不能**因此说「prompt 长度可以随便变」：")
    print(f"      - Qwen3.5 的 max_position_embeddings = 32768，超了就是没训过的区域；")
    print(f"      - 锚点变了意味着绝对相位变了，而权重是在特定范围内拟合的；")
    print(f"      - 官方也明确写了改相机布局/分辨率「not covered by the released weights」。")

    banner("⑦ 真实与 toy 的参数对照")
    print(f"   {'':<28} {'toy':>10} {'真实':>12}")
    print(f"   {'-' * 52}")
    for name, toy, real in (
        ("head_dim", 32, 256),
        ("partial_rotary_factor", 0.25, 0.25),
        ("rotary_dim", ROTARY_DIM, 64),
        ("mrope_section", str(tuple(MROPE_SECTION)), "(11, 11, 10)"),
        ("频率对数", ROTARY_DIM // 2, 32),
        ("rope_theta", "1e7", "1e7"),
    ):
        print(f"   {name:<28} {str(toy):>10} {str(real):>12}")
    print(f"\n   三个值（head_dim、partial_rotary_factor、rope_theta）都必须和 VLM 一致，")
    print(f"   因为它们决定了「同一个位置的旋转矩阵长什么样」。差一个就全乱。")
    print(f"   它们的真实来源不在专家的 config 里，而在 VLM 的 text_config 里。")


if __name__ == "__main__":
    main()
