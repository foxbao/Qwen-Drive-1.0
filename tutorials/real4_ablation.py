"""real4：输入消融 —— 模型到底在看什么（需要 GPU + 完整权重）

【目的】用真实权重做一组**因果干预**：改掉一个输入，看轨迹动多少。

    跑法：python tutorials/real4_ablation.py

【⚠️ 先把免责声明说清楚】这仍然是 **n=1** 的探索性观察：
    - 只有一个满足“历史位姿和速度均非零”的 demo 场景（运行时按条件选择）
    - 只改一个输入，其他都不动，但模型是高度非线性的，**不能把距离当影响力打分**
    - 干预会让输入偏离训练分布（比如把历史置零），这种偏离本身就是一种伤害
    要推广需要多场景、多种子、配对噪声和统计检验。本 stage 只演示**怎么做这个实验**。

【为什么按条件选场景】部分 demo 的 history、ego_velocity 为零，拿它们做历史或速度
    消融没有意义。脚本选择第一个历史位姿和速度均非零的样本；数据集变更时会自动适配。

默认使用 `planner-sft`，因为本实验包含 DIRECT_PLANNING；若改用 `planner-rl`，应把
结果视为分布外的输入干预观察，而不是 planner-rl 的质量结论。
"""

from __future__ import annotations

import dataclasses

from real_common import banner, load_model, load_samples, require_deps

require_deps(require_weights=True, require_planner=True)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from qwen_drive import InferenceMode  # noqa: E402
from qwen_drive.scene import CAMERA_VIEWS  # noqa: E402


def run(model, scene, seed):
    """跑一次 direct planning，返回 (轨迹, 终点)。"""
    out = model.run(InferenceMode.DIRECT_PLANNING, scene=scene, num_samples=1, seed=seed)
    return out.trajectory, out.trajectory[-1, :2]


def compare(name, baseline, baseline_end, trajectory, end, truth):
    """打印相对 baseline 的变化，以及和真值的误差。"""
    shift = np.linalg.norm(trajectory[:, :2] - baseline[:, :2], axis=-1)
    endpoint = float(np.linalg.norm(end - baseline_end))
    num_poses = min(trajectory.shape[0], truth.shape[0])
    ade = float(np.linalg.norm(
        trajectory[:num_poses, :2] - truth[:num_poses, :2], axis=-1).mean())
    print(f"   {name:<28} {shift.mean():>10.3f} {shift.max():>10.3f} "
          f"{endpoint:>11.3f} {ade:>9.3f}")


def main() -> None:
    model = load_model()
    samples = load_samples()
    candidates = [
        sample
        for sample in samples
        if np.linalg.norm(sample.scene.history[:, :2]) > 1e-6
        and np.linalg.norm(sample.scene.ego_velocity) > 1e-6
    ]
    if not candidates:
        raise SystemExit("demo 场景中没有历史位姿和速度均非零的样本，无法运行该消融。")
    sample = candidates[0]
    scene = sample.scene
    truth = sample.future_trajectory
    seed = model.config.noise_seed

    banner("① 场景与基线")
    print(f"   token            {sample.token[:24]}")
    print(f"   历史 x 范围      [{scene.history[:, 0].min():.3f}, {scene.history[:, 0].max():.3f}] m")
    print(f"   自车速度         {float(np.hypot(*scene.ego_velocity)):.3f} m/s")
    print(f"   nav_command      {scene.nav_command}")
    print(f"   真值终点         {np.round(truth[-1, :2], 3).tolist()}")

    baseline, baseline_end = run(model, scene, seed)
    num_poses = min(baseline.shape[0], truth.shape[0])
    baseline_ade = float(np.linalg.norm(
        baseline[:num_poses, :2] - truth[:num_poses, :2], axis=-1).mean())
    print(f"\n   基线 ADE         {baseline_ade:.3f} m")
    print(f"   基线终点         {np.round(baseline_end, 3).tolist()}")

    banner("② 消融结果")
    print(f"   {'干预':<28} {'平均位移':>10} {'最大位移':>10} {'终点位移':>11} {'ADE':>9}")
    print(f"   {'-' * 74}")

    # ── 1) 历史位姿置零 ──────────────────────────────────────────────
    zero_history = dataclasses.replace(
        scene,
        history=np.zeros_like(scene.history),
        history_velocity=np.zeros_like(scene.history_velocity),
        history_acceleration=np.zeros_like(scene.history_acceleration),
    )
    trajectory, end = run(model, zero_history, seed)
    compare("历史位姿/速度/加速度=0", baseline, baseline_end, trajectory, end, truth)

    # ── 2) 只把速度加速度置零，保留位姿 ──────────────────────────────
    zero_dynamics = dataclasses.replace(
        scene,
        history_velocity=np.zeros_like(scene.history_velocity),
        history_acceleration=np.zeros_like(scene.history_acceleration),
    )
    trajectory, end = run(model, zero_dynamics, seed)
    compare("只置零速度/加速度", baseline, baseline_end, trajectory, end, truth)

    # ── 3) 自车状态置零（速度 + 加速度 + driving_command）────────────
    zero_ego = dataclasses.replace(
        scene,
        ego_velocity=np.zeros(2),
        ego_acceleration=np.zeros(2),
        driving_command=np.zeros_like(np.asarray(scene.driving_command)),
    )
    trajectory, end = run(model, zero_ego, seed)
    compare("ego_status 置零", baseline, baseline_end, trajectory, end, truth)

    # ── 4) 换导航指令 ───────────────────────────────────────────────
    other_nav = 2 if scene.nav_command != 2 else 0
    swapped_nav = dataclasses.replace(scene, nav_command=other_nav)
    trajectory, end = run(model, swapped_nav, seed)
    compare(f"nav_command {scene.nav_command}→{other_nav}", baseline, baseline_end,
            trajectory, end, truth)

    # ── 5) 左右相机内容互换（标签不动）──────────────────────────────
    views = {view: list(frames) for view, frames in scene.views.items()}
    views["<FRONT LEFT VIEW>"], views["<FRONT RIGHT VIEW>"] = (
        views["<FRONT RIGHT VIEW>"], views["<FRONT LEFT VIEW>"],
    )
    swapped_views = dataclasses.replace(scene, views=views)
    trajectory, end = run(model, swapped_views, seed)
    compare("左右相机图片互换", baseline, baseline_end, trajectory, end, truth)

    # ── 6) 三路都用正前方（丢掉左右视野）────────────────────────────
    same_views = {view: list(scene.views["<FRONT VIEW>"]) for view in CAMERA_VIEWS}
    single_view = dataclasses.replace(scene, views=same_views)
    trajectory, end = run(model, single_view, seed)
    compare("三路都用 FRONT", baseline, baseline_end, trajectory, end, truth)

    # ── 7) 历史帧替换成当前帧（丢掉时间信息）────────────────────────
    frames = list(scene.views["<FRONT VIEW>"])
    timeline = {
        view: [list(scene.views[view])[-1]] * len(scene.views[view]) for view in CAMERA_VIEWS
    }
    frozen = dataclasses.replace(scene, views=timeline)
    trajectory, end = run(model, frozen, seed)
    compare("历史帧全换成当前帧", baseline, baseline_end, trajectory, end, truth)

    banner("③ 读表之前先看一个陷阱：prompt 文本里另有一份历史")
    print(f"   benchmark 场景的 `instruction_text` 是**存下来的原文**，")
    print(f"   `DrivingScene.instruction()` 遇到它就原样返回，不会再从 history 合成：")
    print()
    for line in scene.instruction().splitlines():
        if line.strip().startswith("-t-"):
            print(f"     {line}")
    print()
    print(f"   也就是说 **历史位姿在输入里有两份**：")
    print(f"     a) prompt 里的**文本**（上面这几行）→ 走 VLM")
    print(f"     b) history 张量 → 走专家的 history_encoder（stage10）")
    print(f"\n   我改 `scene.history` 只动了 (b)，**prompt 文本一个字没变**，")
    print(f"   所以「历史置零」那一行测出来的只是**专家侧那一条路**的影响。")
    print(f"   因此历史置零的位移不能解释为「历史没用」：VLM 仍然看得到完整的历史文本。")
    print(f"\n   ⚠️ 这是 stage10 ④ 那个「冗余入口」问题的升级版：")
    print(f"      有一个入口藏在**提示词文本**里，改张量根本碰不到它。")

    banner("④ 怎么读这张表")
    print(f"   '平均位移' 是整条轨迹相对基线的平均偏差（米）；")
    print(f"   'ADE' 是和**真值**的平均距离 —— 基线本身就有 {baseline_ade:.3f} m 的误差，")
    print(f"   所以 ADE 变小**不代表**干预是「改进」，可能只是碰巧更接近。")
    print(f"\n   三个层次的信号：")
    print(f"     视觉输入（相机互换 / 丢视野 / 冻时间）")
    print(f"     历史张量（位姿 / 速度 / 加速度）—— 注意 ③ 那个陷阱")
    print(f"     元数据（ego_status / nav_command）—— 同样只动了专家侧")

    banner("⑤ ego_status 置零是一种强分布外干预")
    print(f"   表中 `ego_status 置零` 的数值要以本次运行结果为准，不能写死成某台机器的观察。")
    print(f"   这个干预通常很强，因为：")
    print(f"     - 它包含**当前速度**，而这个场景以 ~7 m/s 行驶，")
    print(f"       置零等于告诉模型「车是静止的」→ 规划出一条完全不同的轨迹")
    print(f"     - 它包含 driving_command 的 one-hot，置零后四维全 0，")
    print(f"       而正常输入里恒有一个 1 —— 这是明显的**分布外**输入")
    print(f"\n   所以这一行**不能**读成「ego_status 比图像重要」，只能读成：")
    print(f"     「把速度信息抹掉，可能显著改变规划」—— 这本来就该如此。")
    print(f"\n   想干净地测「导航指令」的影响，应该只改 driving_command 和 nav_command，")
    print(f"   保留速度；而且两个入口一起改（stage10 ④）。")

    banner("⑥ ⚠️ 不能从这张表得出的结论")
    print(f"   1. **不能说「哪个模态更重要」**")
    print(f"      干预幅度不可比：把历史置零是 O(10 m) 的改动，")
    print(f"      nav_command 换一档只是换了一个 one-hot 位置。")
    print(f"      位移更大不等于「更重要」。")
    print(f"\n   2. **不能把「没变化」当成「没用」**")
    print(f"      某个输入改成另一个**同样合理**的取值时，模型完全可以给出")
    print(f"      几乎一样的轨迹 —— 这恰恰说明它泛化得好，不是没在用。")
    print(f"\n   3. **不能忽略冗余入口**")
    print(f"      stage10 ④ 讲过导航相关信息有多条入口。")
    print(f"      只改一个入口，其余入口会把信息补上，效果被稀释。")
    print(f"\n   4. **n=1、单个场景**")
    print(f"      换成 scene 1（左转）或 scene 2（右转），排序可能完全不同。")
    print(f"      要得到可信结论需要：多场景 × 多种子 × 配对噪声 × 统计检验。")

    banner("⑦ 想做得更严谨的话")
    print(f"   - **配对噪声**：先把噪声固定下来，再分别跑 baseline 和干预，")
    print(f"     这样两条轨迹共享同一个起点，（本期教程里 seed 已经固定，但")
    print(f"     real3 ② 说明 bf16 批量差异会带来厘米级抖动，所以要注意噪声底）。")
    print(f"   - **多种子**：每个条件跑 6~8 个种子，看位移的分布而不是单点。")
    print(f"   - **多场景**：至少覆盖 GO STRAIGHT / TURN LEFT / TURN RIGHT 各若干。")
    print(f"   - **真值无关的指标**：ADE 依赖真值，比较「干预前后」时用轨迹位移更干净。")
    print(f"\n   本 stage 的价值在于把**方法**跑通，不在于给出的数字。")


if __name__ == "__main__":
    main()
