# Training Plan

本文档记录在本仓库基础上增加训练能力的实施方案。当前仓库主要提供推理代码和
预训练权重，不包含官方训练脚本、完整训练集或完整训练超参数。因此，本文档中的
第一阶段是一个**可验证的 Planning Expert SFT 实现**，不是对官方训练流程的完整复现。

## 1. 目标与范围

第一版只做以下事情：

- 冻结 Qwen3.5 VLM；
- 使用真实场景图像、历史状态、导航指令和未来轨迹；
- 只训练 `PlanningExpert`；
- 使用 DIRECT planning 路径；
- 使用 clean-endpoint flow-matching loss；
- 保存可以被现有 `load_planner()` 读取的规划专家 checkpoint。

第一版明确不做：

- VLM 全量解冻；
- VQA、感知和规划的联合训练；
- reasoning 文本生成的梯度训练；
- RL、Waymo preference reward 或 NAVSIM 闭环 reward；
- 使用 demo 数据得出模型质量结论。

这样划定范围的原因是：当前推理代码已经有完整的专家网络和数据预处理，但没有训练
入口；而 VLM 联合训练会同时引入激活显存、梯度检查点和多任务数据的问题。

### 当前进度：训练 smoke test

第一步已经实现为一个不依赖真实 VLM 的 CPU smoke test：

```bash
PYTHONPATH=src python scripts/train_planner_smoke.py --steps 80
PYTHONPATH=src python -m unittest discover -s tests -v
```

它使用缩小后的真实 `PlanningExpert` 结构和假的 VLM cache，验证 endpoint flow batch、
有效帧 mask、反向传播、单样本过拟合以及 checkpoint round-trip。这个脚本只验证训练
链路，不代表真实数据或模型质量。

## 2. 当前代码边界

可以直接复用的部分：

- [`QwenDriveProcessor`](../src/qwen_drive/scene.py)：图像 patch 化、ChatML prompt、状态张量；
- [`read_scene_file`](../src/qwen_drive/benchmarks.py)：JSONL 场景读取；
- [`normalize_history`](../src/qwen_drive/trajectory.py)：历史坐标归一化；
- [`PlanningExpert.predict_endpoint`](../src/qwen_drive/planning_expert.py)：可微的专家前向；
- [`QwenDriveForPlanning._scene_cache`](../src/qwen_drive/modeling_qwen_drive.py)：提取 VLM 的
  post-rotary K/V cache。

需要新增或调整的部分：

- 训练用的 VLM prefill，不能复用带 `@torch.no_grad()` 的推理封装；
- flow-matching batch loss；
- Dataset/DataLoader 和 batch collator；
- optimizer、scheduler、AMP、梯度累积和 checkpoint 保存；
- 训练验证指标和 overfit smoke test。

现有的 `generate_trajectory()`、`plan_from_inputs()` 和 `PlanningExpert.sample()` 都是
推理接口，带有 `no_grad` 并且最终返回 NumPy，因此不能直接放进训练循环。

## 3. 第一版数据契约

训练样本使用现有 planning scene JSONL 格式。至少需要：

| 字段 | 形状 | 用途 |
| --- | --- | --- |
| 三路相机、四个时间点 | 12 张图 | VLM prompt 和视觉特征 |
| `hist_traj_10hz` | `[16, 3]` | 专家历史位姿条件 |
| `hist_vel_10hz` | `[16, 2]` | 专家速度条件 |
| `hist_acc_10hz` | `[16, 2]` | 专家加速度条件 |
| `ego_status` | `[8]` | 当前速度、加速度和驾驶指令 |
| `nav_command` | scalar | 导航条件 |
| `future_traj_10hz` | `[50, 3]` | clean endpoint 训练目标 |
| `future_valid_mask_10hz` | `[50]` | 屏蔽无效未来帧 |

必须保持已有数据约定：相机顺序、每帧 `resized_width/resized_height`、prompt 原文、
自车坐标系、10 Hz 时间轴和历史帧顺序都不能自行改变。图像 resize 或 prompt 改动会
改变视觉 token 数和 mRoPE anchor，导致训练输入与发布权重的分布不一致。

`data/demo/` 只包含少量演示场景，适合验证读取、前向和 loss 是否工作，不适合训练出
有意义的驾驶策略。正式训练需要外部构建的 NAVSIM、Waymo 或 PhysicalAI 训练 split，
并按 [`docs/data.md`](data.md) 的格式生成场景文件。

## 4. 第一版训练目标

模型当前是 clean-endpoint 参数化：专家输入 noisy trajectory `x_t`，输出干净未来轨迹
预测 `x1_hat`，而不是直接输出 flow velocity。

第一版建议使用以下训练构造：

```text
x1 = normalize(future_trajectory)
x0 ~ Normal(0, I)
t  ~ Uniform(0, 1)
xt = (1 - t) * x0 + t * x1
x1_hat = PlanningExpert.predict_endpoint(xt, t, ...)
loss = masked_mean((x1_hat - x1) ** 2, future_valid_mask)
```

这里的 `x0`、`t` 分布和损失权重是第一版的工程方案，不应宣称等同于官方训练配方；
官方训练脚本和完整配方没有随仓库发布。训练时仍必须使用配置中的
`trajectory_scale = [165, 25, 1.5703125]`，并沿用现有 heading 表示。

无效未来帧不能参与 loss。对某个样本全部无效的情况，应跳过该样本，而不是用零轨迹
当作监督信号。

## 5. 训练前向路径

第一版每个 batch 的计算顺序如下：

1. DataLoader 读取场景并调用 `QwenDriveProcessor`。
2. 将 `input_ids`、`pixel_values` 和 `image_grid_thw` 放到 GPU。
3. 在冻结且 `eval()` 的 VLM 上执行一次 prefill。
4. 提取 8 份 full-attention post-rotary K/V 和 mRoPE anchor，并 detach。
5. 归一化未来轨迹和历史位姿。
6. 编码 history、velocity、acceleration。
7. 采样 `x0`、`t`，构造 `xt`。
8. 调用 `predict_endpoint()`，保留专家计算图。
9. 使用 `future_valid_mask` 计算 loss，反向更新 Planning Expert。

示意代码：

```python
with torch.no_grad():
    vlm_outputs = model.vlm(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        image_grid_thw=inputs["image_grid_thw"],
        mm_token_type_ids=model._modality_ids(inputs["input_ids"]),
        use_cache=True,
    )
    scene_cache = model._scene_cache(vlm_outputs.past_key_values)
    anchor = model._rope_positions(
        inputs["input_ids"], inputs["image_grid_thw"]
    )[:, :, -1]

history_queries = expert.encode_history(
    normalized_history,
    inputs["nav_command"],
    inputs["history_velocity"],
    inputs["history_acceleration"],
)
prediction = expert.predict_endpoint(
    noisy_trajectory,
    flow_time,
    history_queries,
    scene_cache,
    anchor,
    inputs["nav_command"],
    inputs["ego_status"],
)
```

训练代码不应调用 `sample()`，因为采样循环是推理阶段的 Euler 积分；训练阶段直接监督
单次 endpoint prediction 更便宜，也与现有模型参数化一致。

## 6. 建议的代码结构

第一阶段新增：

```text
src/qwen_drive/training.py       # 训练 forward、flow batch、masked loss
scripts/train_planner.py         # CLI、数据读取、优化器、日志、checkpoint
tests/test_training.py           # 可反向传播、mask、shape、checkpoint 回读
```

目前已落地的是 `src/qwen_drive/training.py`、两个 smoke 脚本、
`scripts/train_planner.py` 和 `tests/test_training.py`。正式 trainer 仍刻意保持
batch size 1，并用 gradient accumulation 形成更大的有效 batch；变长 cache 的通用
批处理留给后续迭代。

真实场景的最小链路也可以单独验证。它会加载完整 VLM，冻结 VLM，只训练 Planning Expert，
并将输出保存成现有 `load_planner()` 可读取的目录：

```bash
PYTHONPATH=src python scripts/train_planner_real_smoke.py \
  --model models/Qwen-Drive-1.0-4B-ms \
  --scenes data/demo/planning_scenes.jsonl \
  --image-archive data/demo/frames.parquet \
  --limit 1 --steps 5 --output outputs/planner-real-smoke
```

该命令需要能加载完整 VLM 的 CUDA/CPU 内存；`data/demo` 只用于验证输入、cache、梯度和
checkpoint 契约，不用于判断规划质量。更完整的训练入口是 `scripts/train_planner.py`，支持
gradient accumulation、可选 validation 和 planner-only resume；正式实验仍需要外部训练
split，并应记录数据版本、模型 revision、seed 和硬件环境。

例如，用同一份 demo 文件跑一个 epoch（仍然只是契约 smoke test）：

```bash
PYTHONPATH=src python scripts/train_planner.py \
  --model models/Qwen-Drive-1.0-4B-ms \
  --scenes data/demo/planning_scenes.jsonl \
  --image-archive data/demo/frames.parquet \
  --limit 1 --epochs 1 --output outputs/planner-demo
```

建议的 `train_planner.py` 参数至少包括：

- `--model`：VLM 权重目录；
- `--scenes`、`--image-root` 或 `--image-archive`；
- `--output`；
- `--epochs`、`--batch-size`、`--gradient-accumulation-steps`；
- `--learning-rate`、`--weight-decay`、`--warmup-steps`；
- `--dtype {bf16,fp32}`；
- `--resume`；
- `--val-scenes`；
- `--seed`。

checkpoint 应至少包含：

- `model.safetensors`，键名与当前 `load_planner()` 兼容；
- `config.json`，保存 expert configuration；
- optimizer、scheduler、epoch、global step 和随机数状态；
- 训练配置和数据文件摘要，便于复现实验。

不要直接调用完整模型的 `save_pretrained()` 作为 planner checkpoint，因为那会把 VLM
也写入输出目录；第一版应单独保存 `planning_expert` 的权重。

## 7. 显存和优化策略

24 GB 显存是当前发布模型的推理建议，不等于训练预算。第一版默认采用：

- VLM 权重冻结、`eval()`、`no_grad()`；
- Planning Expert 使用 bf16 参数和 autocast；
- batch size 从 1 开始；
- gradient accumulation 扩大有效 batch；
- gradient clipping；
- 必要时使用 8-bit Adam 或 Adafactor；
- 先不启用 reasoning generation；
- scene cache 每个 batch 只生成一次，并在 batch 内复用。

如果 1B expert 加上 optimizer state 仍超出显存，再考虑只训练 adapter、LoRA 或分组解冻；
不能默认认为“推理能放进 24 GB”就意味着全量训练也能放进去。

## 8. 验收标准

### 代码级验收

- 单个 batch 的 loss 是有限值；
- `loss.backward()` 后至少一个 expert 参数有非零梯度；
- VLM 参数没有梯度；
- 预测和目标形状都是 `[B, 50, 3]`；
- 全部无效帧样本不会产生 NaN；
- 保存的 planner checkpoint 能被 `load_planner()` 重新加载。

### 过拟合验收

先只用一条样本，固定随机种子运行几十到几百步：

- training loss 应明显下降；
- 同一条件下 endpoint 误差应下降；
- 采样后的 ADE/FDE 应比随机初始化明显改善。

这一步只证明训练链路可学习，不代表模型具有泛化能力。

### 小数据验收

使用一个很小但独立的 train/validation split：

- 每个 epoch 记录 endpoint loss、ADE、FDE 和有效帧比例；
- 对比随机专家、现有 planner checkpoint 和新 checkpoint；
- 用 `scripts/eval_*.py` 评估前，确认预测网格和 `future_valid` 处理一致；
- 固定数据版本、模型版本、seed、dtype、GPU 和 attention backend。

## 9. 后续阶段

### 阶段二：reasoning-conditioned SFT

在 VLM 仍冻结的情况下，使用 reasoning prompt 和生成后的 cache 训练专家。生成文本本身
不参与梯度，训练目标仍是轨迹 endpoint。需要确认训练数据是否包含与 reasoning prompt
匹配的文本分布，不能把 DIRECT 和 REASONING cache 混在一起而不记录模式。

### 阶段三：VLM 参数高效微调

只对 VLM 的视觉/语言模块加入 LoRA 或其他 adapter，保留专家训练路径。此阶段需要：

- 移除 VLM prefill 的 `no_grad`；
- 梯度检查点；
- 明确哪些 full-attention K/V 参与反向；
- 评估显存、吞吐和 VLM 能力退化；
- 同时保留通用 VQA 验证集，防止灾难性遗忘。

### 阶段四：多任务和 RL

感知、VQA 和规划联合训练需要额外的数据、label mapping、任务采样比例和 loss 权重。
RL 还需要可重复的 rollout、Waymo preference 或 NAVSIM PDMS reward；在 SFT trainer
稳定前不建议开始。

## 10. 已知限制

- 官方 SFT 的时间采样、loss weighting、数据混合比例和增强策略未公开在本仓库中；
- 当前本地模型目录只有 `planner-rl`，没有 `planner-sft` warm-start 权重；
- demo 数据不能用于质量结论；
- 训练结果必须区分 open-loop displacement、oracle `minADE` 和 benchmark-specific
  selector；
- 训练期间应固定 prompt 原文、图像 resize 和相机顺序，否则无法与发布模型公平比较。
