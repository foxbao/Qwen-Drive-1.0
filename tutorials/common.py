"""共享积木：stage5 之后各 stage 复用的模块。

包含：常量 + 轨迹归一化 + x 参数化 flow matching + Expert 的全部零件。
用法：`from common import TrajectorySpace, FlowMatchingX, MiniExpert, ...`

对应真实代码：
- TrajectorySpace        ← src/qwen_drive/trajectory.py
- FlowMatchingX          ← src/qwen_drive/planning_expert.py::PlanningExpert.sample
- FourierFeatureEncoder  ← planning_expert.py::FourierFeatureEncoder
- WaypointRotaryEmbedding← planning_expert.py::WaypointRotaryEmbedding
- ExpertLayer            ← planning_expert.py::PlanningExpertLayer
- MiniExpert             ← planning_expert.py::PlanningExpert（缩小版）

【toy 的缩放约定】结构与真实代码逐项对齐，只有宽度被压小。真实 → toy：
    hidden 1024      → 64
    heads/KV-heads   16 / 4   → 4 / 1     （都是 GQA 4:1）
    head_dim 256     → 32
    expert 层数 32   → 8   （layers_per_kv 都是 4，所以 KV 源 8 → 2）
    future points 50 → 50  （不缩，时间轴缩了就看不出轨迹了）
真值一律可以从 config.json 读出，见 README §六。
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# 小张量下多线程是负优化，显式压一下（可用环境变量覆盖）。
torch.set_num_threads(int(os.environ.get("TUTORIAL_NUM_THREADS", "8")))

# ---------------------------------------------------------------- 数据契约
N_WAYPOINTS = 50          # 未来 waypoint 数（5s @ 10Hz）
POINT_DIM = 3             # (x, y, heading)
N_HISTORY = 16            # 历史位姿数
N_HISTORY_QUERY = N_HISTORY - 1   # 丢掉了被当作原点的最老一帧 → 15
NAV_CLASSES = 3           # GO STRAIGHT / TURN LEFT / TURN RIGHT
EGO_STATUS_DIM = 8        # 2 速度 + 2 加速度 + 4 驾驶指令 one-hot
HISTORY_DYNAMICS_DIM = 2  # 速度/加速度每个点 2 维
TRAJ_SCALE = (165.0, 25.0, 1.5703125)
TRAJ_HZ = 10.0

# ---------------------------------------------------------------- 采样器
N_STEPS = 10
MIN_ONE_MINUS_T = 0.1
NOISE_SEED = 42

# ---------------------------------------------------------------- toy 几何
HIDDEN = 64
INTERMEDIATE = 128
NUM_HEADS = 4
NUM_KV_HEADS = 1
HEAD_DIM = 32
PARTIAL_ROTARY_FACTOR = 0.25
ROTARY_DIM = int(HEAD_DIM * PARTIAL_ROTARY_FACTOR)          # 8
MROPE_SECTION = (2, 1, 1)                                    # 和 = ROTARY_DIM/2 = 4
ROPE_THETA = 1.0e7
TIME_EMBED_DIM = 32
TIME_EMBED_SCALE = 1000.0
FOURIER_NUM_FEATURES = 8
FOURIER_MAX_FREQUENCY = 16.0

# layers_per_kv=4：每 4 个 expert 层共用一份 VLM cache
LAYERS_PER_KV = 4
NUM_LAYERS = 8                       # toy：8 层 → 2 份 KV
NUM_KV_SOURCES = NUM_LAYERS // LAYERS_PER_KV

# Qwen3.5 文本塔是混合注意力：每 4 层一个 full_attention，其余是 linear_attention。
# 这是从模型 config 的 layer_types 里数出来的，toy 直接沿用真实值。
VLM_LAYERS = 32
FULL_ATTENTION_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31]

# toy 场景序列长度（真实 prompt 有几千个 token，这里只留个象征值）
SCENE_LEN = 24


def mlp(in_features: int, hidden: int) -> nn.Sequential:
    """真实代码里的 `_mlp`：Linear → SiLU → Linear，两层同宽。"""
    return nn.Sequential(nn.Linear(in_features, hidden), nn.SiLU(), nn.Linear(hidden, hidden))


# ---------------------------------------------------------------- stage3
class TrajectorySpace:
    """(x, y, heading) 的归一化 / 反归一化 / 角度缠绕。

    真实代码把 heading 也塞进同一个除法里，所以 `wrap_heading` 在归一化前后都要做一次：
    除以 scale 不会改变角度落在哪个周期，但反归一化乘回来可能把结果推出 [-pi, pi)。
    """

    def __init__(self, scale=TRAJ_SCALE) -> None:
        self.scale = torch.tensor(scale, dtype=torch.float32)

    @staticmethod
    def wrap_heading(trajectory: torch.Tensor) -> torch.Tensor:
        heading = torch.remainder(trajectory[..., 2:3] + math.pi, 2 * math.pi) - math.pi
        return torch.cat([trajectory[..., :2], heading], dim=-1)

    def normalize(self, trajectory: torch.Tensor) -> torch.Tensor:
        return self.wrap_heading(trajectory) / self.scale.view(1, 1, -1)

    def denormalize(self, trajectory: torch.Tensor) -> torch.Tensor:
        return self.wrap_heading(trajectory * self.scale.view(1, 1, -1))

    def normalize_history(self, history: torch.Tensor) -> torch.Tensor:
        """把历史重新参考到最老的一帧，再归一化并丢掉那一帧。"""
        history = self.wrap_heading(history - history[:, 0:1, :])
        return self.normalize(history[:, 1:, :])


# ---------------------------------------------------------------- stage4
class FlowMatchingX:
    """干净端点（x）参数化的 flow matching 采样器。

    网络预测的是**终点轨迹** x1，不是速度场 v；
    速度由 `v = (x1_hat - x_t) / (1 - t)` 现推，积分走欧拉。
    """

    def __init__(self, n_steps: int = N_STEPS, min_one_minus_t: float = MIN_ONE_MINUS_T) -> None:
        self.n_steps = n_steps
        self.min_one_minus_t = min_one_minus_t

    def sample(self, endpoint_fn, noise: torch.Tensor) -> torch.Tensor:
        """`endpoint_fn(x_t, t) -> x1_hat`，形状全部是 (B, N_WAYPOINTS, 3)。"""
        waypoints = noise.float()
        step = 1.0 / self.n_steps
        for index in range(self.n_steps):
            flow_time = torch.full(
                (waypoints.shape[0],), index * step, dtype=torch.float32
            )
            endpoint = endpoint_fn(waypoints, flow_time)
            # 真实代码就是这一行：remaining 有下限，最后一步不会把预测误差放大。
            remaining = max(1.0 - index * step, self.min_one_minus_t)
            waypoints = waypoints + (endpoint - waypoints) / remaining * step
        return waypoints


# ---------------------------------------------------------------- stage5
class RMSNorm(nn.Module):
    """真实代码在 fp32 里算均方根，再乘回原 dtype。"""

    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


class FourierFeatureEncoder(nn.Module):
    """每个通道一组对数间隔频率的 sin/cos，再过一个 MLP。"""

    def __init__(self, point_dim: int, hidden: int, num_features: int, max_frequency: float) -> None:
        super().__init__()
        self.num_features = num_features
        self.max_frequency = max_frequency
        self.net = mlp(point_dim * num_features * 2, hidden)

    def forward(self, waypoints: torch.Tensor) -> torch.Tensor:
        freqs = torch.logspace(
            0,
            math.log10(self.max_frequency),
            steps=self.num_features,
            device=waypoints.device,
            dtype=waypoints.dtype,
        )
        angles = waypoints.unsqueeze(-1) * freqs * (2 * math.pi)
        features = torch.cat([angles.sin(), angles.cos()], dim=-1).flatten(-2)
        return self.net(features)


class SinusoidalTimeEmbedding(nn.Module):
    """flow 时间 t 的固定正弦编码，真实代码里 scale=1000 把 t∈[0,1] 摊开。"""

    def __init__(self, dim: int, scale: float) -> None:
        super().__init__()
        self.dim = dim
        self.scale = scale

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        decay = math.log(10000) / (half - 1)
        freqs = torch.exp(torch.arange(half, dtype=torch.float32) * -decay)
        angles = self.scale * t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([angles.sin(), angles.cos()], dim=-1)


class WaypointRotaryEmbedding(nn.Module):
    """多段交错 mRoPE，和 VLM 用的是同一套。

    位置 id 形如 [3, B, L]（三个 mRoPE 段各一行），返回 [B, L, 1, rotary_dim] 的 cos/sin。
    """

    def __init__(self, rotary_dim: int = ROTARY_DIM, rope_theta: float = ROPE_THETA,
                 sections=MROPE_SECTION) -> None:
        super().__init__()
        self.rotary_dim = rotary_dim
        self.rope_theta = rope_theta
        self.sections = list(sections)
        if sum(self.sections) != rotary_dim // 2:
            raise ValueError("mrope_section 必须正好等于频率对数 rotary_dim // 2")

    def forward(self, position_ids: torch.Tensor, dtype: torch.dtype):
        exponents = torch.arange(0, self.rotary_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (self.rope_theta ** (exponents / self.rotary_dim))
        angles = position_ids.to(dtype).unsqueeze(-1) * inv_freq.to(dtype)   # [3, B, L, pairs]
        # 三个段交错地取：段 0 占 0,3,6,...；段 1 占 1,4,7,...；段 2 占 2,5,8,...
        merged = angles[0].clone()
        for offset, length in enumerate(self.sections[1:], start=1):
            merged[..., offset : length * 3 : 3] = angles[offset][..., offset : length * 3 : 3]
        emb = torch.cat([merged, merged], dim=-1).unsqueeze(2)               # [B, L, 1, dim]
        return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = torch.chunk(x, 2, dim=-1)
    return torch.cat([-second, first], dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """只旋转前 `cos.shape[-1]` 个通道（partial rotary），剩下原样透传。"""
    rotary_dim = cos.shape[-1]
    rotated, passthrough = x[..., :rotary_dim], x[..., rotary_dim:]
    rotated = rotated * cos + rotate_half(rotated) * sin
    return torch.cat([rotated, passthrough], dim=-1)


class ExpertLayer(nn.Module):
    """一层 diffusion-transformer：联合注意力 + 门控 + SwiGLU，全部受 adaLN 调制。"""

    def __init__(self) -> None:
        super().__init__()
        self.num_heads = NUM_HEADS
        self.num_kv_heads = NUM_KV_HEADS
        self.head_dim = HEAD_DIM
        self.heads_per_group = NUM_HEADS // NUM_KV_HEADS

        # 融合 qkv：每个 KV 组内先放 (query, gate) 再放 key、value。
        qkv_out = NUM_KV_HEADS * (self.heads_per_group * 2 + 2) * HEAD_DIM
        self.input_layernorm = RMSNorm(HIDDEN)
        self.qkv_proj = nn.Linear(HIDDEN, qkv_out, bias=False)
        self.q_norm = RMSNorm(HEAD_DIM)
        self.k_norm = RMSNorm(HEAD_DIM)
        self.o_proj = nn.Linear(NUM_HEADS * HEAD_DIM, HIDDEN, bias=False)

        self.post_attention_layernorm = RMSNorm(HIDDEN)
        self.gate_up_proj = nn.Linear(HIDDEN, INTERMEDIATE * 2, bias=False)
        self.down_proj = nn.Linear(INTERMEDIATE, HIDDEN, bias=False)

        # AdaLN-Zero：输出层零初始化，训练第一步整层等于恒等映射。
        self.adaln_modulation = nn.Sequential(nn.SiLU(), nn.Linear(HIDDEN, 6 * HIDDEN))
        nn.init.zeros_(self.adaln_modulation[1].weight)
        nn.init.zeros_(self.adaln_modulation[1].bias)

    def split_qkv(self, fused: torch.Tensor):
        batch, length, _ = fused.shape
        head_dim, groups, per_group = HEAD_DIM, NUM_KV_HEADS, self.heads_per_group
        fused = fused.view(batch, length, groups, (per_group * 2 + 2) * head_dim)
        gated_query, key, value = torch.split(
            fused, [per_group * 2 * head_dim, head_dim, head_dim], dim=3
        )
        query, gate = torch.chunk(gated_query, 2, dim=-1)
        return (
            query.reshape(batch, length, self.num_heads, head_dim),
            gate.reshape(batch, length, self.num_heads, head_dim),
            key.reshape(batch, length, groups, head_dim),
            value.reshape(batch, length, groups, head_dim),
        )

    def forward(self, hidden_states, scene_key, scene_value, cos, sin, condition):
        batch, length, _ = hidden_states.shape
        shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = (
            m.unsqueeze(1) for m in self.adaln_modulation(condition).chunk(6, dim=-1)
        )

        residual = hidden_states
        x = self.input_layernorm(hidden_states) * (1 + scale_attn) + shift_attn
        query, gate, key, value = self.split_qkv(self.qkv_proj(x))
        query = apply_rotary(self.q_norm(query), cos, sin)
        key = apply_rotary(self.k_norm(key), cos, sin)

        # 联合注意力：waypoint 自己的 K/V 直接接在场景 cache 后面。
        attn = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            torch.cat([scene_key, key], dim=1).transpose(1, 2),
            torch.cat([scene_value, value], dim=1).transpose(1, 2),
            enable_gqa=self.num_heads != self.num_kv_heads,
        ).transpose(1, 2).reshape(batch, length, -1)
        attn = attn * torch.sigmoid(gate.reshape(batch, length, -1))   # 每头一个输出门
        hidden_states = residual + (1 + gate_attn) * self.o_proj(attn)

        residual = hidden_states
        x = self.post_attention_layernorm(hidden_states) * (1 + scale_ffn) + shift_ffn
        swiglu_gate, swiglu_up = self.gate_up_proj(x).chunk(2, dim=-1)
        return residual + (1 + gate_ffn) * self.down_proj(F.silu(swiglu_gate) * swiglu_up)


class MiniExpert(nn.Module):
    """Qwen-Drive Planning Expert 的缩小版，逐项对应真实 PlanningExpert。"""

    def __init__(self, scene_len: int = SCENE_LEN) -> None:
        super().__init__()
        self.scene_len = scene_len
        self.trajectory_proj = nn.Linear(POINT_DIM, HIDDEN)
        self.fourier_encoder = FourierFeatureEncoder(
            POINT_DIM, HIDDEN, FOURIER_NUM_FEATURES, FOURIER_MAX_FREQUENCY
        )
        self.waypoint_embed = nn.Embedding(N_WAYPOINTS, HIDDEN)
        self.time_embed = SinusoidalTimeEmbedding(TIME_EMBED_DIM, TIME_EMBED_SCALE)
        self.time_mlp = mlp(TIME_EMBED_DIM, HIDDEN)
        self.nav_mlp = mlp(NAV_CLASSES, HIDDEN)
        self.ego_mlp = mlp(EGO_STATUS_DIM, HIDDEN)

        history_dim = N_HISTORY_QUERY * POINT_DIM + NAV_CLASSES
        dynamics_dim = N_HISTORY * HISTORY_DYNAMICS_DIM
        self.history_encoder = mlp(history_dim, HIDDEN)
        self.history_velocity_encoder = mlp(dynamics_dim, HIDDEN)
        self.history_acceleration_encoder = mlp(dynamics_dim, HIDDEN)
        self.query_fusion = mlp(HIDDEN * 7, HIDDEN)

        self.rotary_emb = WaypointRotaryEmbedding()
        self.layers = nn.ModuleList(ExpertLayer() for _ in range(NUM_LAYERS))
        self.final_layernorm = RMSNorm(HIDDEN)
        self.out_proj = nn.Linear(HIDDEN, POINT_DIM)

    # ---- 位置锚点 ----
    def waypoint_positions(self, anchor: torch.Tensor, length: int) -> torch.Tensor:
        """anchor 形状 [3, B]：每个 mRoPE 段一行，waypoint 从 anchor+1 依次排下去。"""
        steps = torch.arange(1, length + 1, dtype=anchor.dtype)
        return anchor.unsqueeze(-1) + steps

    # ---- 条件 ----
    @staticmethod
    def one_hot(index: torch.Tensor, num_classes: int, dtype: torch.dtype) -> torch.Tensor:
        valid = (index >= 0) & (index < num_classes)
        onehot = F.one_hot(index.clamp(0, num_classes - 1).long(), num_classes).to(dtype)
        return onehot * valid.to(dtype).unsqueeze(-1)

    def encode_history(self, history, nav_command, velocity, acceleration):
        dtype = history.dtype
        batch = history.shape[0]
        nav_onehot = self.one_hot(nav_command, NAV_CLASSES, dtype)
        pose_query = self.history_encoder(
            torch.cat([history.reshape(batch, -1), nav_onehot], dim=-1)
        )
        velocity_query = self.history_velocity_encoder(velocity.reshape(batch, -1))
        acceleration_query = self.history_acceleration_encoder(acceleration.reshape(batch, -1))
        return pose_query, velocity_query, acceleration_query

    # ---- 单步去噪：预测干净端点 ----
    def predict_endpoint(self, waypoints, flow_time, history_queries, scene_cache,
                         position_anchor, nav_command, ego_status):
        dtype = waypoints.dtype
        batch, length, _ = waypoints.shape
        pose_query, velocity_query, acceleration_query = history_queries

        time_condition = self.time_mlp(self.time_embed(flow_time).to(dtype))
        waypoint_index = torch.arange(length)
        # 七路信号拼起来 → 一个 MLP 融合成 waypoint token。
        broadcast = [
            self.trajectory_proj(waypoints),
            self.fourier_encoder(waypoints),
            time_condition.unsqueeze(1).expand(-1, length, -1),
            pose_query.unsqueeze(1).expand(-1, length, -1),
            self.waypoint_embed(waypoint_index).unsqueeze(0).expand(batch, -1, -1),
            velocity_query.unsqueeze(1).expand(-1, length, -1),
            acceleration_query.unsqueeze(1).expand(-1, length, -1),
        ]
        hidden_states = self.query_fusion(torch.cat(broadcast, dim=-1))

        nav_onehot = self.one_hot(nav_command, NAV_CLASSES, dtype)
        condition = time_condition + self.nav_mlp(nav_onehot) + self.ego_mlp(ego_status)

        cos, sin = self.rotary_emb(self.waypoint_positions(position_anchor, length), dtype)
        for index, layer in enumerate(self.layers):
            scene_key, scene_value = scene_cache[index // LAYERS_PER_KV]
            hidden_states = layer(
                hidden_states,
                scene_key.expand(batch, -1, -1, -1),
                scene_value.expand(batch, -1, -1, -1),
                cos, sin, condition,
            )
        return self.out_proj(self.final_layernorm(hidden_states)).float()

    def sample(self, scene_cache, position_anchor, history, history_velocity,
               history_acceleration, nav_command, ego_status, noise,
               n_steps: int = N_STEPS):
        history_queries = self.encode_history(
            history, nav_command, history_velocity, history_acceleration
        )

        def endpoint_fn(x, t):
            return self.predict_endpoint(
                x, t, history_queries, scene_cache, position_anchor, nav_command, ego_status
            )

        return FlowMatchingX(n_steps=n_steps).sample(endpoint_fn, noise)


# ---------------------------------------------------------------- 假 VLM
def fake_scene_cache(num_kv_sources: int = NUM_KV_SOURCES, batch: int = 1,
                     scene_len: int = SCENE_LEN, seed: int = 0):
    """造一份形状正确的「VLM 场景 cache」。

    真实代码里这是 VLM 分组查询注意力层的旋转后 K/V，形状 [B, S, kv_heads, head_dim]，
    每层一对，只取 full_attention 层。toy 用随机张量代替，讲解形状和拓扑足够。
    """
    generator = torch.Generator().manual_seed(seed)
    return [
        (
            torch.randn(batch, scene_len, NUM_KV_HEADS, HEAD_DIM, generator=generator),
            torch.randn(batch, scene_len, NUM_KV_HEADS, HEAD_DIM, generator=generator),
        )
        for _ in range(num_kv_sources)
    ]


def fake_batch(batch: int = 2, seed: int = 0):
    """造一组合法形状的专家输入，供各 stage 直接跑。

    注意 `history` 已经是**归一化、重参考过**的 15 个点：
    `TrajectorySpace.normalize_history` 把最老一帧当原点并丢掉它，
    所以专家看到的是 15 而不是 16（真实代码同样如此，见 docs/model.md）。
    """
    generator = torch.Generator().manual_seed(seed)
    history = torch.randn(batch, N_HISTORY_QUERY, POINT_DIM, generator=generator) * 0.5
    return {
        "history": history,
        "history_velocity": torch.randn(batch, N_HISTORY, HISTORY_DYNAMICS_DIM, generator=generator),
        "history_acceleration": torch.randn(
            batch, N_HISTORY, HISTORY_DYNAMICS_DIM, generator=generator
        ),
        "nav_command": torch.randint(0, NAV_CLASSES, (batch,), generator=generator),
        "ego_status": torch.randn(batch, EGO_STATUS_DIM, generator=generator),
    }
