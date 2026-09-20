# Qwen-Drive 1.0 学习笔记

读 Qwen-Drive-1.0 的推理代码，理解「一个 VLM 同时做感知、问答、规划」是怎么搭起来的。

> ## ⚠️ 定位说明（重要）
>
> **本教程是「带 toy 复现的源码导读」，不是一个可复现的模型实现。**
> 每个 stage 用最小代码解释**一个**机制，最后（§六）再把 toy 模块映射回真实文件。
>
> toy 的规模比真实小 3 个数量级：真实 VLM 是 Qwen3.5-4B（2560 隐层、32 层、24.8 万词表），
> 真实专家是 ~1.0 B 参数（1024 隐层、32 层）；toy 专家约 70 万参数。
> 两者有**逐项的结构对应**，但不是等比例缩小。
>
> 本教程面向**已经学过 `alpamayo1.5/tutorials` 的读者**，
> 所以凡是 Alpamayo 已经讲透的（flow matching、KV cache 条件化、CoC、多相机）
> 都只做对照，不重复推导；重点放在**Qwen-Drive 不一样的地方**。

---

## 一、Qwen-Drive 和 Alpamayo 差在哪（先看这张表）

两者都是「VLM + 独立轨迹生成头」，但**几乎每个具体选择都不一样**：

| | Alpamayo 1.5 | Qwen-Drive 1.0 |
|---|---|---|
| 动作空间 | 单轮车动作 `(加速度, 曲率)`，要积分 | **直接回归 `(x, y, heading)` 航点**，无积分 |
| 专家的条件化 | cross-attention | **联合注意力** `attn(Q_wp, [K_scene ; K_wp], …)` |
| 条件走哪进 | 全塞进 K/V | **场景走注意力，元数据走 adaLN** |
| 生成目标 | 速度场 `v = x1 - x0` | **干净端点 `x1`**（速度现推 `(x1-x)/(1-t)`） |
| 位置编码 | 学习/傅里叶 | **继承 VLM 的 mRoPE，接在 prompt 后面** |
| 任务面 | 只有规划 | **VQA + 3D 感知 + 规划，共享一个 VLM** |
| VLM 注意力 | 全注意力 | **混合**：24 层线性 + 8 层真注意力 |
| 读几层 cache | 逐层 | **8 份，每 4 个专家层共用一份** |

**一句话心智模型**：

> VLM 读图 + 读 prompt（prompt 里印着历史位姿和导航指令）→ 一次性产出逐层 K/V cache
> → 专家把 50 个航点 token 接在这份 cache **后面**，用同一个注意力和同一套 RoPE 去噪
> → 10 步欧拉积分出一条约 5 秒的未来轨迹。

**三个「一次」**（决定开销结构）：
- **prompt 只读一次**：prefill 一次，cache 全程复用，`num_samples` 不增加 VLM 开销
- **图只编码一次**：视觉 token 是 prompt 的一部分，不在专家里重复出现
- **专家的 50 个航点是一次性并行出的**，不是自回归，所以注意力 **不 causal**

---

## 二、整体架构（一图流）

```
输入（两条并行的路）
├─ 图片序列 3 路 × 4 帧 ───────────────► 拼进 prompt（定长占位块）
└─ 历史 / ego_status / nav_command ──┬─► 也印成文本进 prompt（另一份拷贝！）
                                    └─► 张量，绕过 VLM 直接进专家

  ① VLM prefill（Qwen3.5-4B）
     ├─ 混合注意力 32 层，其中 8 层 full_attention、24 层 linear_attention
     └─ 取出 **8 层真注意力**的 post-rotary K/V ──► scene_cache（8 份）
        以及 prompt 最后一个 token 的 mRoPE 位置 ──► anchor [3, B]

  ② 专家去噪（循环 10 次，Qwen-Drive 版）
     x_t (50, 3) ─trajectory_proj / fourier─┐
     flow 时间 t ───────────────────────────┤
     历史位姿/速度/加速度（各一个 MLP）──────┤ 七路 concat → MLP → 50 个航点 token
     航点序号 embedding ────────────────────┤
     位置 = anchor+1 … anchor+50 ───────────┘
        │
        ├─► 32 层，每层：联合注意力(读 [8 份场景 cache ; 自己的 K/V]) + adaLN + SwiGLU
        └─► out_proj ─► 干净端点 x1_hat
     x ← x + (x1_hat − x) / max(1−t, 0.1) × dt

  ③ 反归一化
     归一化轨迹 ×[165, 25, 1.5703125] ─► (x, y, heading) 米/弧度，自车当前帧，10 Hz
```

**别混的数字**：
- **50** = 未来航点数（5 s @ 10 Hz）
- **8** = 专家能读的 VLM cache 份数（= 32 层 / layers_per_kv 4）
- **32** = 专家层数，也是 VLM 层数（两者巧合都是 32，含义不同）
- **3385 / 522** = `data/demo` scene 0、DIRECT 模式的 prompt 长度 / mRoPE 锚点示例；换场景或模式会变化（见 §五 坑 1）

---

## 三、Stage 一览

下面先列规划主线；Stage 12（感知）和 Stage 13（评测）是完成主线后按需进入的独立分支。

| # | 文件 | 核心概念 | 学到什么 |
|---|---|---|---|
| 0 | `stage0_framework.py` | 数据流 + 形状 | 四个阶段、三条输入路径、`(1,50,3)` 从哪来 |
| 1 | `stage1_scene.py` | 场景与坐标系 | x 前 / **y 左** / heading 左正；两个「指令」的区别；16→15 帧的重参考 |
| 2 | `stage2_prompt.py` | Prompt 组装 | demo scene 0 中 12 张图 = 3054 个占位符；DIRECT 与 REASONING 的末尾结构不同 |
| 3 | `stage3_trajectory.py` | 归一化 | `[165, 25, 1.5703125]` 的来历；heading 缠绕；历史换原点 |
| 4 | `stage4_flow_x.py` | **x 参数化采样** | 为什么预测终点；10 步欧拉**无离散误差**；`min_one_minus_t` 何时生效 |
| 5 | `stage5_expert_block.py` | 一层解剖 | 融合 qkv + 输出门 + 逐头 qk-norm + 部分旋转 + **adaLN-Zero** |
| 6 | `stage6_joint_attention.py` | **联合注意力** | waypoint K/V 与 scene K/V 拼接；toy 因果探针 |
| 7 | `stage7_rope_anchor.py` | 位置锚点 | anchor 随当前 prompt 动态计算（demo scene 0 为 522）；图像按「格」计费；mRoPE 交错填充 |
| 8 | `stage8_hybrid_kv.py` | 混合注意力 | 为什么 32 层专家只读 8 份 cache；哪些参数被 VLM 锁死 |
| 9 | `stage9_reasoning_cache.py` | cache 补全 | generate 停在终止符；为什么要补 `<\|im_end\|>`；锚点怎么跟着挪 |
| 10 | `stage10_conditioning.py` | 七路融合 + adaLN | 航点 token 的构造；导航相关信息的三条入口；adaLN-Zero 的实测证据 |
| 11 | `stage11_modes.py` | 三种模式与采样 | VQA / DIRECT / REASONING；种子方案；开销结构 |
| 12 | `stage12_perception.py`（可选） | 感知分支概览 | 两个特征抽头；UVTR 体素池化；三个头的输出；bf16-only 的坑 |
| 13 | `stage13_eval.py`（可选） | 评测 | 输入契约、三套网格转换；toy 曲线展示两种 Waymo 约定的差异；min 系列是 oracle |

### real 系列（跑真实模型）

| 文件 | 需要 GPU | 内容 |
|---|---|---|
| `real1_input_trace.py` | 否* | Stage 2 后运行：真数据 + 真 tokenizer；不加载 checkpoint，但会从 config 构造随机 VLM |
| `real2_inference_trace.py` | 是 | Stage 11 后运行：三种模式真跑；cache 形状；**num_samples 开销曲线** |
| `real3_determinism.py` | 是 | real2 后运行：同 seed 复现；**批大小会改变结果**；best-of-N |
| `real4_ablation.py` | 是 | Stage 10/11 后运行：输入消融；**prompt 里另有历史拷贝**这个陷阱 |

### 依赖关系

| 过渡 | 改了什么 |
|---|---|
| 0→1→2→3 | 输入侧：场景 → prompt → 归一化。都是「模型看到什么」 |
| 3→4 | 采样器：网络预测什么、怎么积分 |
| 4→5→6→7→8 | 专家内部：单层 → 联合注意力 → 位置 → KV 来源 |
| 8→9→10→11 | 组装：推理模式的 cache → 条件构造 → 三种模式的完整路径 |
| 11→12 | 可选分支：共享 VLM 后接感知头；**不是规划前置** |
| 11→13 | 可选下游：把规划输出转换到 benchmark 网格并计算指标 |

### 推荐学习路径

```text
规划主线：0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11
                         ├─ real1（Stage 2 后，真实输入对账）
                         ├─ real2 → real3（Stage 11 后，真实推理与确定性）
                         └─ real4（Stage 10/11 后，输入消融）

独立分支：11 → 12（感知概览）
          11 → 13（评测与指标契约）
```

每个 stage 至少应回答三件事：输入/输出 shape 是什么、对应的真实源码在哪里、运行后哪条
assertion 或数字能证明本章结论。Stage 0 只固定接口和数据流，不验证真实数值、heading
缠绕或 attention；这些行为由后续 stage 分别验证。

---

## 四、关键概念速查

**坐标系**：自车当前帧，`x` 前、`y` **左**、`heading` 左正；米 / 弧度；10 Hz。
Demo 里 nav_command 和真值终点方向对得上（`stage1` ④ 可自证）。

**两条指令**（别混）：
- `nav_command`：整数 0/1/2，`(GO STRAIGHT, TURN LEFT, TURN RIGHT)`，给专家的 `nav_mlp`
- `driving_command`：4 维 one-hot，在 `ego_status` 里，给 adaLN 条件

**归一化**：`trajectory_scale = [165, 25, 1.5703125]`。
第三项是 **`pi/2` 的 bfloat16 取整值** —— 不是笔误，改了就偏移。

**采样器**（一句话）：
```
x ← x + (x1_hat − x) / max(1 − t, 0.1) × dt     共 10 步
```
预测的是**终点**不是速度。分母下限只在 `n_steps > 10` 时生效。

**专家读什么**：VLM 8 个 `full_attention` 层的 **post-rotary** K/V，
每 4 个专家层共用一份。linear_attention 层没有逐 token K/V，读不了。

**三种模式**：VQA（不碰专家）/ DIRECT（turn 在 prompt 里闭合）/
REASONING（先生成理由，再补全 cache 后规划）。

本页中的 `3385`、`3054`、`522` 和 `1.3 m` 都不是模型的通用常量：前三个来自
`data/demo` 的 scene 0（DIRECT），最后一个来自 `stage13_eval.py` 的合成轨迹。真实
benchmark 应以脚本运行时打印的 shape、位置和指标为准。

---

## 五、踩过的坑（实测发现，不是读文档读出来的）

### 坑 1：mRoPE 锚点是 522，不是 prompt 长度 3385

prompt 有 3385 个 token，但 `get_rope_index` 给出的最后一个位置是 **522**。

因为 mRoPE 对图像**按网格计费**：一张 patch 网格 (26, 24) 的历史帧有 156 个 token，
但它只占 `(26/2, 24/2) = (13, 12)` 的位置格 —— **merge_size=2 的效果**。

```
纯文本 token 331 + 9 张历史帧 ×13 + 3 张当前帧 ×25 = 523；实测最后位置为 522，
因为位置下标从 0 开始。
```

**影响**：航点拿到的是位置 `523 … 572`，不是 `3386 … 3435`。
如果用「token 序号」当位置，会差 6 倍多。验证见 `real1` ④⑤。

### 坑 2：批大小会改变轨迹 —— 和 docs 说的不一样

初始噪声逐位相同，但在 bf16 下完整轨迹可能不逐位相同；本机一次实测最大差约 8 cm。
这是硬件、CUDA、attention backend 和 batch 形状相关的数值现象，不是通用误差保证。

原因是 bf16 的批量矩阵乘随 batch 改变规约顺序，舍入差在 32 层 × 10 步里被放大。
量级比 ADE（~0.4 m）小一个数量级，不影响结论方向，但**位级复现需要固定 batch**。
数据见 `real3` ②。

### 坑 3：prompt 文本里另有一份历史，消融消不掉

benchmark 场景的 `instruction_text` 是**存下来的原文**，里面印着历史位姿。
改 `scene.history` 只影响**专家张量那一路**，prompt 文本一个字没变。

所以「把历史置零」测出来的位移只有 0.02 m，**不代表历史没用**，
只代表「VLM 那边还看得到完整的历史文本」。见 `real4` ③。

这是 `stage10` ④「导航相关信息有三条入口」的升级版：**有的入口藏在提示词里**。

### 坑 4：adaLN-Zero 让条件通路在初始化时是死的

`adaln_modulation` 的最后一层被零初始化，所以随机权重下
`condition` 无论是什么，6 个调制量恒为 0。

实测：置零 `ego_status`（它**只**走 adaLN）→ 输出变化 **0.00%**。
把 adaLN 权重唤醒后 → 立刻变成 13.4%。见 `stage10` ⑤。

### 坑 5：`min_one_minus_t` 只在步数 > 10 时生效

默认 10 步时，最后一步 `1-t = 0.1` 恰好等于下限，**下限不生效**，欧拉精确落点。
步数 > 10 时最后一步只跳一部分，产生**系统性欠冲**（`n_steps=100` 时 0.117）。

所以「加步数更准」在这里**不成立**。见 `stage4` ②。

### 坑 6：Waymo 的两种重采样约定在 toy 曲线上会有明显差异

当前 toy 曲线的最大 gap 约为 1.3 m；这不是 benchmark 的固定偏差。真实预测应按目标
benchmark 的网格重新计算，并在比较 displacement 数字前确认是否使用
`--official-4hz-grid`。见 `stage13` ④。

### 坑 7：感知分支的 CUDA kernel 要求 bf16

发布的 `multi_scale_deformable_attn_cuda` 在 CUDA 上遇到非 bf16 会抛 `TypeError`；
CPU/off-GPU 路径有 torch fallback，适合功能验证但性能和支持范围不同。视锥网格**故意
不注册成 buffer**，以免被 `.to(bfloat16)` 转掉精度。见 `stage12` ⑥。

---

## 六、Toy → Real 对照

| 概念 | toy | 真实 | 真实值出处 |
|---|---|---|---|
| 专家隐层 | 64 | **1024** | `config.json` |
| 专家层数 | 8 | **32** | 同上 |
| 注意力头 / KV 头 | 4 / 1 | **16 / 4** | 同上（GQA 都是 4:1） |
| head_dim | 32 | **256** | 同上 |
| partial rotary | 0.25 → 8 | **0.25 → 64** | 同上 |
| mrope_section | (2,1,1) | **(11,11,10)** | 同上（和都 = rotary_dim/2） |
| layers_per_kv | 4 | **4** | 同上 |
| → KV 源数 | 2 | **8** | 32 层 VLM / interval 4 |
| 航点数 | 50 | **50** | 不缩 |
| 场景 token 数 | 24 | **3385**（demo scene 0 DIRECT；其中 3054 是视觉） | `real1` 实测 |
| 位置锚点 | 由 stage7 手填 | **522**（同一 demo scene/mode） | `real1` 实测 |
| 专家参数量 | ~0.7 M | **~1.0 B** | `stage5` ⑦ 逐层累加 |

**验算 1.0 B**：单层 31,981,568 × 32 = 1,023,410,176。
`planner-rl/model.safetensors` 是 2.08 GB ÷ 2 字节(bf16) ≈ 1.04 B ✓

### 关键结构对应（务必对照看）

| 真实文件 | toy 对应 |
|---|---|
| `planning_expert.py::PlanningExpertLayer` | `common.ExpertLayer` |
| `planning_expert.py::PlanningExpert.predict_endpoint` | `common.MiniExpert.predict_endpoint` |
| `planning_expert.py::WaypointRotaryEmbedding` | `common.WaypointRotaryEmbedding` |
| `planning_expert.py::FourierFeatureEncoder` | `common.FourierFeatureEncoder` |
| `planning_expert.py::PlanningExpert.sample` | `common.FlowMatchingX.sample` |
| `trajectory.py::normalize_history` 等 | `common.TrajectorySpace` |
| `planning_expert.py::_mlp` | `common.mlp` |

### 逐项差距清单（toy 没有的东西）

1. **非方阵的注意力**：真实 `attention_hidden = 16×256 = 4096`，是 `hidden=1024` 的 **4 倍**；
   toy 是 2 倍。`o_proj` 的压缩比不同。
2. **bfloat16 全程**：toy 全是 float32。真实的 bf16 及其舍入是**权重拟合的一部分**
   （`FourierFeatureEncoder` 甚至显式重建频率表来复现 bf16 的取整）。
3. **线性注意力层**：toy 只把 `full_attention_layers` 当常量用，没有真的实现
   `linear_attention`。stage8 只讲「为什么读不了」，不讲它怎么算。
4. **CoC 文本生成**：toy 的 stage9 用假 token 序列演示 cache 补全，
   没有跑真的 `vlm.generate`。real2/real3 会跑真的。
5. **感知分支**：stage12 只做形状追踪和一个玩具体素池化，没有 BEVFormer 的
   可变形注意力实现。

---

## 七、环境与运行

### CPU 部分（规划主线、可选分支、real1）

以下命令都从仓库根目录执行。Toy stage 需要 `torch`、`numpy`；使用真实 tokenizer 的
stage1/2/13 和 `real1` 还需要 `transformers`、`Pillow`。`real1` 不加载 9 GB checkpoint，
但会从 `config.vlm_config` 构造随机 VLM 来调用 `get_rope_index`，因此需要完整的
transformers 模型类和较高 CPU 内存。模型目录可用 `QWEN_DRIVE_MODEL_DIR` 覆盖。

```bash
export PYTHONPATH=src
for n in $(seq 0 11); do
  python "$(printf 'tutorials/stage%d_' "$n")"*.py
done
python tutorials/stage12_perception.py  # 可选：感知分支概览
python tutorials/stage13_eval.py        # 可选：评测分支
# 需要 config/tokenizer；不需要 model.safetensors
python tutorials/real1_input_trace.py
```

### GPU 部分（real2 ~ real4）

需要 CUDA 和约 24 GB 显存（本机实测占用约 11 GB）。使用当前环境时先激活对应
虚拟环境，命令不要写死某台机器的绝对路径：

```bash
python tutorials/real2_inference_trace.py
```

| 项 | 值 |
|---|---|
| torch | `pyproject.toml` 要求 `>=2.8.0` |
| transformers | `pyproject.toml` 要求 `>=5.14.0,<5.15.0` |
| `attn_implementation` | `"sdpa"`（无需安装 flash-attn；可显式切换） |
| 权重目录 | `QWEN_DRIVE_MODEL_DIR` 指向完整 VLM 目录 |
| 加载耗时 / 单次规划 | 取决于 GPU、驱动、权重位置和 attention backend |

> `real1` 只需要模型目录中的 `config.json` 和 `tokenizer.json`。`real2`--`real4` 还需要
> 根目录的 `model.safetensors` 和规划专家权重；可以通过 `QWEN_DRIVE_MODEL_DIR`、
> `QWEN_DRIVE_PLANNER_DIR` 指定路径。真实教程在同时存在两个专家目录时默认选
> `planner-sft`，因为它覆盖 DIRECT 和 REASONING；`planner-rl` 只应在
> `REASONING_PLANNING` 下使用。

---

## 八、读完这几件事值得自己动手

1. **把 `real4` 的消融做扎实**：加场景（4 个 demo 都用上）、加种子（每条件 6~8 个）、
   用配对噪声。现在的表只能说明方法，不能说明结论。
2. **验证坑 3**：改 `instruction_text` 里的历史数字（而不是改张量），
   看轨迹动多少 —— 那才是「历史真正的影响」。
3. **读感知代码的原始出处**：`src/qwen_drive_perception/` 里的
   `bev_encoder.py`（BEVFormer 编码器）和 `view_transform.py`（UVTR）。
   stage12 只给了地图，没给细节。
4. **对照论文**：https://arxiv.org/abs/2609.00111 —— 特别是分阶段训练策略那部分，
   本教程完全没讲**训练**（只讲推理）。

---

## 附：本教程的写法约定

- **toy 一律 CPU、float32、小张量**，结构对齐、宽度压小
- 每个 stage 的 docstring 都写了【目的】【和 Alpamayo 的差异】【简化/注意】
- **凡是估算都标注了**「方向确定、数字别当真」；凡是实测都给出了复现命令
- 遇到和官方文档冲突的地方（坑 2），**写明冲突并以实测为准**
- 不把「示意图」包装成「实验」：stage6 的因果探针用真实层跑，
  没跑的就明确说是示意（如 stage12 的体素池化只是形状演示）
