"""Stage 9：推理模式的 cache 补全 —— 一个五行的、但必须做对的小把戏

【目的】REASONING_PLANNING 模式下，VLM 先生成一句理由，专家再读「生成后」的 cache。
听起来只是「先跑一段再跑下一段」，但有一个**位置对齐**问题必须手工补上，
否则航点拿到的 mRoPE 位置就偏离训练时的位置（stage7 说位置来自锚点）。

真实代码 `modeling_qwen_drive._prefill_with_reasoning`：

    ① 生成：vlm.generate(..., eos_token_id=[im_end, eos])   ← 遇到 im_end 就停
    ② 截断：content = new_ids[: 第一个终止符的位置]           ← 丢掉 runaway 的后半段
    ③ 补全：closed_turn = content + [im_end] + newline
    ④ 只把「cache 里还没有的」补进去：pending = closed_turn[already_cached:]
    ⑤ 锚点：anchor = prompt_anchor + len(closed_turn)

【为什么必须补】`generate` 在采样到 <|im_end|> 的那一刻停下，**这个 token 本身
没有进 cache**（它只出现在返回值里）。但训练时，航点 token 是接在一个
**已经闭合的完整 turn** 后面的。少了 im_end + "\\n" 这两个 token，
航点的位置就会往前错 2 格，attention 的相位跟着错。

【和 Alpamayo 的差异】Alpamayo 的 CoC 是「把 CoC 文本拼回 prompt 再前向一次」，
位置自然正确。Qwen-Drive 走的是 **KV cache 复用** 路线：为了省掉一次完整前向，
必须手工把 cache 补到训练时的长度。省了算力，多了这个坑。
"""

from __future__ import annotations


def banner(title: str) -> None:
    print("\n" + "=" * 66)
    print(title)
    print("=" * 66)


class FakeCache:
    """模拟 transformers 的 Cache：记录已缓存的长度，可增量追加。"""

    def __init__(self) -> None:
        self.length = 0
        self.tokens: list[str] = []

    def append(self, tokens: list[str]) -> None:
        self.tokens.extend(tokens)
        self.length += len(tokens)

    def get_seq_length(self) -> int:
        return self.length


def main() -> None:
    PROMPT_LEN = 3400
    IM_END, NEWLINE = "<|im_end|>", "\\n"

    banner("① 生成停在哪里：终止符被采样出来了，但它不在 cache 里")
    cache = FakeCache()
    cache.append([f"prompt_{i}" for i in range(PROMPT_LEN)])
    print(f"   prefill 之后 cache 长度          {cache.get_seq_length()}")

    # 模拟 generate：模型逐步吐出 token，最后吐出 <|im_end|>
    generated = ["Accelerate", " through", " the", " intersection", IM_END, " and", " then", " blah"]
    #                                            ↑ 第一个终止符在这里
    print(f"\n   generate 返回的 new_ids（去掉 prompt 后）:")
    for index, token in enumerate(generated):
        mark = "  ← 第一个终止符" if index == 4 else ""
        print(f"     [{index}] {token!r}{mark}")

    banner("② 截断：终止符之后的内容是 runaway，必须丢掉")
    content = generated
    for position, token in enumerate(generated):
        if token == IM_END:
            content = generated[:position]
            break
    print(f"   content（实际理由）            {content}")
    print(f"   被丢掉的部分                   {generated[4:]}")
    print(f"\n   真实代码注释：「Anything after it is runaway generation and must not reach")
    print(f"   the cache, or the turn ending would appear twice.」")
    print(f"\n   注意两个不同的东西：")
    print(f"     **cache** 保留 generate 吐出的**全部** token（含终止符），专家要读这个")
    print(f"     交给用户的**文本**只是 content，且还会再剥一层 thinking 块")

    banner("③ 补全：闭合 turn，回到训练时的形态")
    closed_turn = content + [IM_END] + [NEWLINE]
    print(f"   closed_turn = content + [<|im_end|>] + [\\n]")
    print(f"   长度 = {len(content)} + 1 + 1 = {len(closed_turn)}")
    print(f"\n   训练时航点就是接在这样一个**闭合的 assistant turn** 后面的。")
    print(f"   DIRECT 模式里那个 turn 在 prompt 里就已经闭合了（stage2 ②），")
    print(f"   REASONING 模式里必须在这里手工闭合 —— 两条路殊途同归。")

    banner("④ 增量补：只补 cache 里还没有的那几个")
    already_cached = cache.get_seq_length() - PROMPT_LEN
    pending = closed_turn[already_cached:]
    print(f"   cache 里已有（generate 留下的）  {already_cached} 个 token")
    print(f"   closed_turn 需要                  {len(closed_turn)} 个 token")
    print(f"   pending（要补进去的）             {pending}")
    print(f"\n   为什么要算这个差？因为 generate 已经把它们生成的部分塞进 cache 了。")
    print(f"   直接整段重放会**重复**，序列就长了。真实代码用 cache_position 精确续写：")
    print(f"     self.vlm(input_ids=torch.tensor([pending]),")
    print(f"              past_key_values=cache,")
    print(f"              cache_position=torch.arange(")
    print(f"                  prompt_length + already_cached,")
    print(f"                  prompt_length + len(closed_turn)))")

    banner("⑤ 锚点：位置要跟着一起挪")
    prompt_anchor = 100        # toy 起点；真实实现必须从当前 prompt 动态计算
    anchor = prompt_anchor + len(closed_turn)
    print(f"   toy prompt 的锚点                prompt_anchor          = {prompt_anchor}")
    print(f"   REASONING 模式的锚点            prompt_anchor + len(closed_turn)")
    print(f"                                 = {prompt_anchor} + {len(closed_turn)} = {anchor}")
    print(f"\n   `anchor = prompt_anchor + len(closed_turn)` —— **一行，但是必须的**。")
    print(f"   len(closed_turn) 里已经包含了生成出来的理由，所以这条式子")
    print(f"   算的就是「闭合 turn 最后一个 token 的位置」。")
    print(f"   如果忘了加，航点位置会停在 prompt 结尾，差 {len(closed_turn)} 格 → 相位错位。")
    print(f"   注意：不能把 stage7 scene 0 的 522 复制到这里；REASONING prompt 的 anchor 要重算。")
    print(f"\n   ⚠️ 一个更细的点：如果模型生成时**刹不住**（过了 im_end 还继续吐），")
    print(f"   cache 里会多出几个 runaway token，它们的 K/V 也在 cache 里。")
    print(f"   但锚点只按 `len(closed_turn)` 算 —— 也就是**故意按「turn 在第一个终止符处闭合」")
    print(f"   来定位置**，因为那才是训练时的形态。多出来的 token 不参与位置记账。")

    banner("⑥ 两条路径对比")
    print(f"   {'':<22} {'DIRECT_PLANNING':>24} {'REASONING_PLANNING':>26}")
    print(f"   {'-' * 74}")
    rows = [
        ("user turn", "原样", "末尾多一句请给推理"),
        ("assistant turn", "prompt 里已闭合", "留空，等生成"),
        ("VLM 生成", "无", "贪心解码 ≤ max_reasoning_tokens"),
        ("cache 来源", "一次 prefill", "prefill + 生成 + 补全"),
        ("锚点", "prompt 末尾位置", "prompt 末尾 + 已生成长度"),
        ("专家输入", "同一份 cache", "含模型自己写的理由"),
    ]
    for name, direct, reasoning in rows:
        print(f"   {name:<22} {direct:>24} {reasoning:>26}")
    print(f"\n   两种模式**共用同一个专家**，差别只在喂进去的 cache 长什么样。")
    print(f"   这也是为什么同一份权重能同时支持「不想考」和「想考」两种用法。")

    banner("⑦ 为什么值得这么麻烦")
    print(f"   朴素做法：把理由文本拼回 prompt，重跑一次完整前向。")
    print(f"     代价 = 再算一次 3385+ 个 token 的 prefill")
    print(f"   真实做法：生成时顺手留下 cache，再补 2 个 token。")
    print(f"     代价 = 2 个 token 的前向")
    print(f"\n   差了三个数量级。而代价就是上面这一堆「必须做对」的记账。")
    print(f"   推论：REASONING 模式比 DIRECT 模式贵的主要是**生成那段文本**，")
    print(f"   不是多了一次视觉编码 —— 图只编码了一次。")


if __name__ == "__main__":
    main()
