"""Stage 10：条件化 —— 七路 query 融合 + 三条 adaLN 条件

【目的】专家每层要回答两个问题：「这个航点 token 长什么样」和「这一整个场景整体
该怎么去噪」。Qwen-Drive 把这两个问题分给两套完全不同的机制：

    ① **逐 token** 的构造（每层都一样，只在最前面算一次）
       七路信号 concat → MLP → waypoint token
         1. 当前噪声航点 x_t          trajectory_proj(3 → 1024)
         2. 它的 Fourier 特征         fourier_encoder(3 → 1024)
         3. flow 时间 t               time_mlp(time_embed(t))
         4. 历史位姿编码（15×3 + nav one-hot）
         5. 航点序号 embedding         waypoint_embed(50 → 1024)
         6. 历史速度编码（16×2）
         7. 历史加速度编码（16×2）
       七路各 1024 维 → concat 成 7168 → query_fusion MLP → 1024

    ② **逐层** 的条件（每层都注入一次，通过 adaLN）
       condition = time_condition + nav_mlp(nav_onehot) + ego_mlp(ego_status)
       三个 1024 维向量**相加**，再产生 6 个调制量（stage5 ⑥）

【和 Alpamayo 的差异】Alpamayo 的 Expert 把条件全部塞进 cross-attention 的 K/V。
Qwen-Drive 里 K/V **只放场景**，其余信息走两条侧路：一条进 token 构造，一条进 adaLN。
换句话说，**场景走注意力，元数据走仿射调制**。

【一个值得注意的冗余】导航相关信息有三条入口（见 ④）：
    nav_onehot(3) → history_encoder 的输入
    nav_onehot(3) → nav_mlp → condition
    driving_command(4) → ego_status → ego_mlp → condition
前两条是 `nav_command` 的不同入口，第三条是独立的 `driving_command`，并非同一个
张量。多条入口意味着网络有多个机会利用它，也意味着**做消融时不能只关一个口**。
"""

from __future__ import annotations

import torch

from common import (
    EGO_STATUS_DIM,
    HIDDEN,
    HISTORY_DYNAMICS_DIM,
    N_HISTORY,
    N_HISTORY_QUERY,
    N_WAYPOINTS,
    NAV_CLASSES,
    POINT_DIM,
    MiniExpert,
)


def banner(title: str) -> None:
    print("\n" + "=" * 66)
    print(title)
    print("=" * 66)


def main() -> None:
    torch.manual_seed(0)
    expert = MiniExpert()
    expert.eval()

    batch = 2
    history = torch.randn(batch, N_HISTORY_QUERY, POINT_DIM)
    velocity = torch.randn(batch, N_HISTORY, HISTORY_DYNAMICS_DIM)
    acceleration = torch.randn(batch, N_HISTORY, HISTORY_DYNAMICS_DIM)
    nav_command = torch.tensor([1, 2])[:batch]
    ego_status = torch.randn(batch, EGO_STATUS_DIM)
    waypoints = torch.randn(batch, N_WAYPOINTS, POINT_DIM)
    flow_time = torch.full((batch,), 0.3)

    banner("① 三条历史编码器：位姿 / 速度 / 加速度各走各的")
    nav_onehot = expert.one_hot(nav_command, NAV_CLASSES, torch.float32)
    pose_input = torch.cat([history.reshape(batch, -1), nav_onehot], dim=-1)
    history_dim = N_HISTORY_QUERY * POINT_DIM + NAV_CLASSES
    dynamics_dim = N_HISTORY * HISTORY_DYNAMICS_DIM
    print(f"   {'编码器':<26} {'输入形状':>18} {'展开维度':>10} {'输出':>10}")
    print(f"   {'-' * 68}")
    print(f"   {'history_encoder':<26} {str(tuple(pose_input.shape)):>18} "
          f"{history_dim:>10} {HIDDEN:>10}")
    print(f"   {'  ├─ 历史位姿':<26} {str(tuple(history.shape)):>18} "
          f"{N_HISTORY_QUERY * POINT_DIM:>10} {'':>10}")
    print(f"   {'  └─ nav one-hot':<26} {str(tuple(nav_onehot.shape)):>18} "
          f"{NAV_CLASSES:>10} {'':>10}")
    print(f"   {'history_velocity_encoder':<26} {str(tuple(velocity.shape)):>18} "
          f"{dynamics_dim:>10} {HIDDEN:>10}")
    print(f"   {'history_acceleration_enc':<26} {str(tuple(acceleration.shape)):>18} "
          f"{dynamics_dim:>10} {HIDDEN:>10}")
    print(f"\n   注意三者都是 `reshape(batch, -1)` 拍平后过 MLP —— **没有时序建模**。")
    print(f"   1.5 s 的时序换成了「长度固定的向量」，靠 stage5 的 waypoint 序号 embedding")
    print(f"   和 RoPE 去补位置感。这和 VLM 那侧（历史是图片序列）不是一回事。")

    banner("② 七路融合：waypoint token 是怎么长出来的")
    time_condition = expert.time_mlp(expert.time_embed(flow_time))
    queries = expert.encode_history(history, nav_command, velocity, acceleration)
    expanded = [
        expert.trajectory_proj(waypoints),
        expert.fourier_encoder(waypoints),
        time_condition.unsqueeze(1).expand(-1, N_WAYPOINTS, -1),
        queries[0].unsqueeze(1).expand(-1, N_WAYPOINTS, -1),
        expert.waypoint_embed(torch.arange(N_WAYPOINTS)).unsqueeze(0).expand(batch, -1, -1),
        queries[1].unsqueeze(1).expand(-1, N_WAYPOINTS, -1),
        queries[2].unsqueeze(1).expand(-1, N_WAYPOINTS, -1),
    ]
    names = [
        "① 噪声航点 x_t",
        "② Fourier(x_t)",
        "③ flow 时间 t",
        "④ 历史位姿(+nav)",
        "⑤ 航点序号",
        "⑥ 历史速度",
        "⑦ 历史加速度",
    ]
    print(f"   {'#':>16} {'来源形状':>22} {'广播后':>22} 说明")
    print(f"   {'-' * 84}")
    raw_shapes = [
        "(B, 50, 3)", "(B, 50, 3)", "(B,)", "(B, 15, 3)", "(50,)", "(B, 16, 2)", "(B, 16, 2)"
    ]
    for name, source, tensor in zip(names, raw_shapes, expanded):  # noqa: B007
        print(f"   {name:>16} {source:>22} {str(tuple(tensor.shape)):>22}")
    fused = torch.cat(expanded, dim=-1)
    print(f"\n   concat 之后             {tuple(fused.shape)}   7 × {HIDDEN} = {7 * HIDDEN}")
    print(f"   query_fusion MLP        {7 * HIDDEN} → {HIDDEN} → {HIDDEN}")
    hidden = expert.query_fusion(fused)
    print(f"   进去专家前              {tuple(hidden.shape)}")

    banner("③ 三条 adaLN 条件是相加的，不是拼接的")
    condition = (
        time_condition
        + expert.nav_mlp(nav_onehot)
        + expert.ego_mlp(ego_status)
    )
    print(f"   time_condition          {tuple(time_condition.shape)}   来自 flow 时间")
    print(f"   nav_mlp(nav_onehot)     {tuple(expert.nav_mlp(nav_onehot).shape)}   来自导航指令")
    print(f"   ego_mlp(ego_status)     {tuple(expert.ego_mlp(ego_status).shape)}   来自 {EGO_STATUS_DIM} 维自车状态")
    print(f"   ─────────────────────────────────────")
    print(f"   condition（相加）        {tuple(condition.shape)}")
    print(f"\n   相加而不是拼接：三者都被投影到**同一个 {HIDDEN} 维空间**再叠加，")
    print(f"   网络只能在这个共享空间里「叠加语义」，不能靠「哪一段是哪个」来区分。")
    print(f"   好处是每个来源都能独立影响全部 6 个调制量，坏处是**不可解释**：")
    print(f"   没法说「第 37 维是导航指令」。")

    banner("④ 导航相关信息有三条入口 —— 做消融时要区分")
    print(f"   {'入口':<34} {'形状':>10}   {'去处':<20}")
    print(f"   {'-' * 70}")
    print(f"   {'nav_onehot（拼进 history_encoder）':<34} {'(B, 3)':>10}   {'逐 token 的第 ④ 路':<20}")
    print(f"   {'nav_mlp(nav_onehot)':<34} {'(B, 3)':>10}   {'adaLN condition':<20}")
    print(f"   {'ego_status 里的 driving_command':<34} {'(B, 4)':>10}   {'ego_mlp → condition':<20}")
    print(f"\n   `nav_command` 是 3 类整数，`driving_command` 是 4 维 one-hot —— ")
    print(f"   **两套不同的编码**（stage1 ③ 讲过），但语义上是同一件事。")
    print(f"\n   ⚠️ 所以「把导航指令置零」这种消融，如果不把三个入口一起改，")
    print(f"   测出来的不是「模型有多依赖导航」，而是「剩下的冗余入口有多够用」。")

    banner("⑤ 因果探针：置零每一路，看输出变多少 —— 结果会暴露一个设计")
    print(f"   做法：正常跑一次 predict_endpoint，再依次把某一路置零，比较输出差异。")
    print(f"   权重是**随机初始化**的（没加载 checkpoint），只看接线的有无。")
    print()
    from common import HEAD_DIM, NUM_KV_HEADS, NUM_KV_SOURCES, SCENE_LEN

    scene_cache = [
        (
            torch.randn(1, SCENE_LEN, NUM_KV_HEADS, HEAD_DIM),
            torch.randn(1, SCENE_LEN, NUM_KV_HEADS, HEAD_DIM),
        )
        for _ in range(NUM_KV_SOURCES)
    ]
    anchor = torch.zeros(3, batch)
    # encode_history 只吃这四样；ego_status 是独立走 adaLN 的。
    base_args = {
        "history": history, "velocity": velocity, "acceleration": acceleration,
        "nav_command": nav_command,
    }

    def run_with(ego=ego_status, **override):
        arguments = dict(base_args)
        arguments.update(override)
        with torch.no_grad():
            return expert.predict_endpoint(
                waypoints, flow_time,
                expert.encode_history(**arguments),
                scene_cache, anchor, arguments["nav_command"], ego,
            )

    base = run_with()
    probes = [
        ("history（历史位姿）", dict(history=torch.zeros_like(history))),
        ("velocity（历史速度）", dict(velocity=torch.zeros_like(velocity))),
        ("acceleration（加速度）", dict(acceleration=torch.zeros_like(acceleration))),
        ("ego_status → 全 0", dict(ego=torch.zeros_like(ego_status))),
    ]

    def report(title: str) -> None:
        print(f"   {title}")
        print(f"   {'置零的输入':<24} {'相对变化':>10}")
        print(f"   {'-' * 40}")
        for name, override in probes:
            out = run_with(**override)
            delta = (out - base).abs().max().item()
            relative = delta / max(base.abs().max().item(), 1e-9)
            print(f"   {name:<24} {relative:>9.2%}")
        print()

    report("── 第一次：原样的 MiniExpert ──")
    print(f"   前三路都明显有影响 —— 它们走的是**逐 token 的七路融合**，线路通了。")
    print(f"   但 ego_status 是 **0.00%**：它对输出毫无影响。")
    print()
    print(f"   这不是接错了线。回忆 stage5 ⑥：adaLN 调制层的输出被**零初始化**：")
    print(f"       adaln_modulation = Sequential(SiLU(), Linear(hidden, 6*hidden))")
    print(f"       nn.init.zeros_(adaln_modulation[1].weight / .bias)")
    print(f"   所以 `condition` 算出来是任意值，经过零权重层之后都变成 0，")
    print(f"   6 个调制量全是 0 —— **整层的条件通路在初始化时是死的**。")
    print(f"   ego_status 只走 adaLN 这一条路（不像 nav 还兼走七路融合），")
    print(f"   于是它的影响**恰好精确为 0**。这反过来验证了 AdaLN-Zero 确实生效。")
    print()

    # 把 adaLN 的零权重换成小的随机值，条件通路就"活"了
    import copy
    awake = copy.deepcopy(expert)
    awake.eval()
    for layer in awake.layers:
        with torch.no_grad():
            layer.adaln_modulation[1].weight.normal_(0, 0.02)
            layer.adaln_modulation[1].bias.normal_(0, 0.02)

    def run_awake(**override):
        # "ego" 走 adaLN，不进 encode_history，先摘出来。
        ego = override.pop("ego", ego_status)
        arguments = dict(base_args)
        arguments.update(override)
        with torch.no_grad():
            return awake.predict_endpoint(
                waypoints, flow_time,
                awake.encode_history(**arguments),
                scene_cache, anchor, arguments["nav_command"], ego,
            )

    report_base = run_awake()
    print(f"── 第二次：把每层的 adaLN 权重从 0 改成小随机值 ──")
    print(f"   {'置零的输入':<24} {'相对变化':>10}")
    print(f"   {'-' * 40}")
    for name, override in probes:
        out = run_awake(**override)
        delta = (out - report_base).abs().max().item()
        relative = delta / max(report_base.abs().max().item(), 1e-9)
        print(f"   {name:<24} {relative:>9.2%}")
    print()
    print(f"   把零初始化「唤醒」之后，ego_status 立刻有了影响 —— 条件通路是通的，")
    print(f"   只是训练**开始时**它被刻意关掉了，让梯度从一个恒等的安全点长出来。")
    print()
    print(f"   ⚠️ 这些数字只说明接线正确与否，**不代表真实模型里各输入的相对重要性**。")
    print(f"      随机权重下每一路都被用到；训练后某些路可能几乎被忽略。")
    print(f"      要谈「模型到底在看什么」，必须在**加载真实权重**后重做（见 real4）。")

    banner("⑥ 一个小细节：velocity/acceleration 用满 16 帧，位姿只用 15 帧")
    print(f"   history              → normalize_history 丢掉最老一帧 → {N_HISTORY_QUERY} 帧")
    print(f"   history_velocity     → 不重参考，**保持 {N_HISTORY} 帧**")
    print(f"   history_acceleration → 同上，{N_HISTORY} 帧")
    print(f"\n   因为速度和加速度是**微分量的序列**，不依赖原点在哪儿，")
    print(f"   不需要（也不应该）做重参考。位置会被减去最老一帧影响，速度不会。")
    print(f"   代码上体现为 `_history(trajectory, 'hist_vel', 16)` 之后直接进 MLP，")
    print(f"   而 history 要先过 `normalize_history`。")


if __name__ == "__main__":
    main()
