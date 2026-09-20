"""Stage 0：骨架 —— 先把「模块接口 + 数据流」钉死，再往里填实现

【目的】建立 Qwen-Drive 的整体心智模型。本 stage **不做任何真实计算**，只跑通形状：
    ① 预处理：DrivingScene(图 + 历史 + 指令) ─► input_ids + pixel_values + 专家侧张量
    ② 场景编码：VLM 前向一次 ─► scene_cache（逐层 K/V）+ position_anchor
    ③ 扩散采样：噪声 x ─Expert(读 scene_cache)─► 干净端点 x1_hat ─► 欧拉步 ─► x
    ④ 反归一化：归一化轨迹 ─► (x, y, heading) 米/弧度

【和 Alpamayo 的结构差异（这是本教程的主线）】
  1. 动作空间不是 (加速度, 曲率)，而是**直接在自车坐标系下的 (x, y, heading) 航点**。
     于是没有「动作→轨迹」的积分步，只有归一化 / 反归一化。
  2. 条件化不是 cross-attention，而是**联合注意力**：
     `attn(Q_wp, [K_scene ; K_wp], [V_scene ; V_wp])`，waypoint token 和场景 token 在同一段
     注意力里。专家本身**没有 condition 参数**，场景是顺着 K/V 进来的（stage6 详讲）。
  3. waypoint token 的位置不是学出来的，而是**接着 VLM 的 mRoPE 位置往下排**
     （anchor+1 ...），见 stage7。

【简化】所有模块都是【空壳】，forward 一律返回 zeros / 随机张量：
  - Processor 不真的切图、不真的 tokenize，只按真实公式算 token 数
  - VLM 不读图，直接返回 zeros 的逐层 K/V
  - Expert 不做 attention，直接返回 zeros 的端点
  - 采样循环的 10 步是真的，只有 step_fn 是假的
  真实规模：VLM Qwen3.5-4B（2560 隐层、32 层）+ Expert 1.0B（1024 隐层、32 层）。
  本 stage 的 toy 张量宽度全部压到 64，见 README §六。
"""

from __future__ import annotations

import torch
import torch.nn as nn

# ---- 超参数：全部来自 models/Qwen-Drive-1.0-4B/config.json ----
BATCH = 2                 # 同时规划几帧场景
N_WAYPOINTS = 50          # 未来 50 个点（5s @ 10Hz）
POINT_DIM = 3             # (x, y, heading)
N_HISTORY = 16            # 历史 16 帧（1.5s @ 10Hz），最后一帧是当前
N_HISTORY_QUERY = 15      # 重参考后丢掉最老一帧 → 15
N_CAMERAS = 3             # FRONT / FRONT LEFT / FRONT RIGHT
N_FRAMES_PER_CAM = 4      # 每路相机 4 个时刻
EGO_STATUS_DIM = 8        # 2 速度 + 2 加速度 + 4 驾驶指令
NAV_CLASSES = 3           # GO STRAIGHT / TURN LEFT / TURN RIGHT

# expert 几何（真实值，toy 只在 token 宽度上做缩水，这里仍按真值展示）
EXPERT_LAYERS = 32
LAYERS_PER_KV = 4
NUM_KV_SOURCES = EXPERT_LAYERS // LAYERS_PER_KV   # = 8
HEAD_DIM = 256
NUM_KV_HEADS = 4
HIDDEN = 64               # ← toy 缩水位；真实是 1024
SCENE_LEN = 24            # ← toy 缩水位；真实 prompt 有几千个 token

# 每个当前帧的 token 数：896x512 图，patch 16，merge 2 → (512/16)x(896/16)/4
CURRENT_TOKENS = (512 // 16) * (896 // 16) // 4   # 448


class Processor:
    """真实类是 ``qwen_drive.scene.QwenDriveProcessor``。

    真实实现做两件事：把每张图 patchify 成 (token 数, patch 维度)，以及把 ChatML prompt
    的 id 拼出来（视图标签 + frame 标签 + vision_start/image_token×N/vision_end）。
    这里只算 token 数，不真的处理像素。
    """

    def encode(self, scene: dict) -> dict:
        num_images = N_CAMERAS * N_FRAMES_PER_CAM
        return {
            "input_ids": torch.zeros(BATCH, SCENE_LEN, dtype=torch.long),
            "pixel_values": torch.zeros(
                BATCH, num_images * CURRENT_TOKENS, 3 * 2 * 16 * 16
            ),
            "num_image_tokens": num_images * CURRENT_TOKENS,
            "history": torch.zeros(BATCH, N_HISTORY_QUERY, POINT_DIM),
            "ego_status": torch.zeros(BATCH, EGO_STATUS_DIM),
            "nav_command": torch.zeros(BATCH, dtype=torch.long),
        }


class VLM(nn.Module):
    """真实类是 ``transformers`` 里的 ``Qwen3_5ForConditionalGeneration``。

    关键：本 stage 的 VLM 不输出 hidden_states，而是输出**逐层的 K/V cache**——
    这就是专家要读的东西。真实代码只取 ``full_attention`` 层的 K/V（32 层里有 8 层），
    因为混合注意力的 linear_attention 层根本不维护可读的 KV cache（stage8 详讲）。
    """

    def forward(self, inputs: dict):
        batch = inputs["input_ids"].shape[0]
        # 每层一份 (key, value)，形状 [B, S, kv_heads, head_dim]
        scene_cache = [
            (
                torch.zeros(batch, SCENE_LEN, NUM_KV_HEADS, HEAD_DIM),
                torch.zeros(batch, SCENE_LEN, NUM_KV_HEADS, HEAD_DIM),
            )
            for _ in range(NUM_KV_SOURCES)
        ]
        # 位置锚点：prompt 最后一个 token 的 mRoPE 位置，三个段各一个数
        anchor = torch.zeros(3, batch)
        return scene_cache, anchor


class PlanningExpert(nn.Module):
    """真实类是 ``qwen_drive.planning_expert.PlanningExpert``。

    它不是「动作空间」，而是**一步去噪器**：吃当前噪声轨迹 + flow 时间，吐干净端点预测。
    """

    def forward(self, waypoints, flow_time, scene_cache, anchor):
        # 真实实现在这里做 32 层联合注意力，本 stage 返回零端点
        return torch.zeros_like(waypoints)


class FlowMatchingX:
    """干净端点（x）参数化的欧拉采样器。真实实现在 ``PlanningExpert.sample`` 里。"""

    def __init__(self, n_steps: int = 10, min_one_minus_t: float = 0.1) -> None:
        self.n_steps = n_steps
        self.min_one_minus_t = min_one_minus_t

    def sample(self, endpoint_fn, noise: torch.Tensor) -> torch.Tensor:
        x = noise.float()
        step = 1.0 / self.n_steps
        for index in range(self.n_steps):
            t = torch.full((x.shape[0],), index * step)
            endpoint = endpoint_fn(x, t)
            # 除以「剩余时间」，下限 min_one_minus_t 防止最后一步放大误差
            remaining = max(1.0 - index * step, self.min_one_minus_t)
            x = x + (endpoint - x) / remaining * step
        return x


class TrajectorySpace:
    """真实实现在 ``qwen_drive.trajectory``：除以 scale 归一化，乘回来反归一化。"""

    SCALE = torch.tensor([165.0, 25.0, 1.5703125])

    def denormalize(self, trajectory: torch.Tensor) -> torch.Tensor:
        return trajectory * self.SCALE.view(1, 1, -1)


class QwenDrive(nn.Module):
    """把一切串起来。真实类是 ``qwen_drive.modeling_qwen_drive.QwenDriveForPlanning``。"""

    def __init__(self) -> None:
        super().__init__()
        self.vlm = VLM()
        self.expert = PlanningExpert()
        self.sampler = FlowMatchingX()
        self.processor = Processor()
        self.space = TrajectorySpace()

    def _scene_cache(self, inputs):
        return self.vlm(inputs)

    def generate_trajectory(self, scene: dict, num_samples: int = 1) -> torch.Tensor:
        inputs = self.processor.encode(scene)                 # ① 预处理
        scene_cache, anchor = self._scene_cache(inputs)       # ② VLM 前向，拿 cache

        # ③ 扩散采样：每个样本一份独立噪声，全部共享同一份 scene cache
        noise = torch.randn(num_samples, N_WAYPOINTS, POINT_DIM)

        def endpoint_fn(x, t):
            return self.expert(x, t, scene_cache, anchor)

        normalized = self.sampler.sample(endpoint_fn, noise)
        return self.space.denormalize(normalized)             # ④ 反归一化


if __name__ == "__main__":
    print("=" * 62)
    print("Stage 0：Qwen-Drive 数据流与张量维度自检")
    print("=" * 62)

    scene = {"token": "demo", "nav_command": 0}
    model = QwenDrive()
    inputs = model.processor.encode(scene)

    print(f"\n① 预处理（Processor）")
    print(f"   input_ids            {tuple(inputs['input_ids'].shape)}    (B, prompt 长度)")
    print(f"   pixel_values         {tuple(inputs['pixel_values'].shape)}")
    print(f"   ├─ 图像数            {N_CAMERAS} 路 × {N_FRAMES_PER_CAM} 帧 = {N_CAMERAS * N_FRAMES_PER_CAM}")
    print(f"   └─ 单帧 token 数     {CURRENT_TOKENS}  = (512/16)×(896/16)/2²")
    print(f"   history              {tuple(inputs['history'].shape)}    (B, 15, 3) ← 16 帧丢掉最老一帧")
    print(f"   ego_status           {tuple(inputs['ego_status'].shape)}     (B, 8)")
    print(f"   nav_command          {tuple(inputs['nav_command'].shape)}      (B,)")

    scene_cache, anchor = model._scene_cache(inputs)
    print(f"\n② VLM 前向（只取 full_attention 层的 KV）")
    print(f"   scene_cache          长度 {len(scene_cache)}  ← 32 层 expert 每 4 层共用一份")
    print(f"   单份 K 形状          {tuple(scene_cache[0][0].shape)}  = (B, S, {NUM_KV_HEADS} 个 KV 头, {HEAD_DIM})")
    print(f"   position_anchor      {tuple(anchor.shape)}   = (3 个 mRoPE 段, B)")

    x0 = torch.randn(BATCH, N_WAYPOINTS, POINT_DIM)
    v = torch.zeros(BATCH, N_WAYPOINTS, POINT_DIM)
    t = torch.tensor([0.3, 0.3])
    print(f"\n③ 扩散一步的内部数据流")
    print(f"   噪声轨迹 x_t         {tuple(x0.shape)}")
    print(f"   flow 时间 t          {tuple(t.shape)}")
    print(f"   端点预测 x1_hat      {tuple(v.shape)}   ← 注意预测的是终点，不是速度")
    remaining = 1.0 - 0.3
    print(f"   推导速度 v           (x1_hat - x_t)/{remaining:.1f}  → {tuple(v.shape)}")

    traj = model.generate_trajectory(scene, num_samples=3)
    print(f"\n④ 完整采样（num_samples=3）")
    print(f"   采样轨迹（归一化）    (3, {N_WAYPOINTS}, {POINT_DIM})   内部单位，除以 scale 之前")
    print(f"   反归一化后轨迹        {tuple(traj.shape)}    (num_samples, 50, 3)")
    print(f"   输出单位             米 / 弧度，自车当前帧坐标系，10 Hz")
    print(f"   注：本 stage 的专家是空壳，所以轨迹全是 0；只看形状。")

    print(f"\n⑤ 三种推理模式（详见 stage11）")
    for mode, desc in (
        ("VQA", "只用 VLM 回答问题，专家不参与"),
        ("DIRECT_PLANNING", "user turn 不提推理要求，assistant turn 直接空着"),
        ("REASONING_PLANNING", "VLM 先写一句理由，专家读「生成后」的 cache 来规划"),
    ):
        print(f"   {mode:20s} {desc}")
