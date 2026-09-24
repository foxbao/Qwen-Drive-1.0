# VLM LoRA / 多任务训练

## 当前实现

`scripts/train_vlm_lora_multitask.py` 使用一份共享的 Qwen-Drive VLM LoRA，在三个任务之间
交替训练：

| 任务 | 监督 | 更新参数 |
| --- | --- | --- |
| NAVSIM | 10 Hz 未来轨迹的 masked endpoint flow loss | VLM LoRA + Planning Expert |
| DriveLM | 驾驶场景问答 assistant answer token CE | VLM LoRA |
| A-OKVQA | COCO 图片选择题 answer token CE | VLM LoRA |

每个 optimizer update 默认只抽一个任务，三个任务权重相同时按约 1:1:1 采样；NAVSIM 的
reasoning-conditioned pilot 已有独立机制，但当前联合 LoRA 的 NAVSIM 路径仍然是 direct prompt，
不生成/监督 reasoning 文本。DriveLM 按 scene 做稳定的 5% holdout，A-OKVQA 使用官方 val，
避免近似相同帧跨到训练与验证两边。VQA loss 只计算 assistant 答案 token，不计算问题和图像
token。训练脚本会在反向传播时检查 LoRA 梯度；NAVSIM 还会检查 Planning Expert 梯度。

数据不需要额外转换；adapter 直接读现有 NAVSIM JSONL、DriveLM JSON 和 A-OKVQA JSON/COCO 图片。
首次运行会检查图像路径完整性，数据较大时这一步会花一点时间。

## 依赖与快速 smoke test

`qwen-drive` 环境需要 PEFT；仓库提供可选依赖：

```bash
pip install -e '.[lora]'
```

确认配置中的模型/数据路径可用后，先运行三任务 smoke test。它确实会各跑一次 NAVSIM、DriveLM、
A-OKVQA 的前向与反向，但不会保存大型 checkpoint：

```bash
MAX_STEPS=3 NAVSIM_TRAIN_LIMIT=4 NAVSIM_VAL_LIMIT=2 \
  VAL_SAMPLES_PER_TASK=1 TASK_SEQUENCE=navsim,drivelm,aokvqa \
  NO_SAVE_CHECKPOINT=1 OUTPUT_DIR=outputs/vlm-lora-smoke \
  bash scripts/run_vlm_lora_multitask_train.sh
```

配置在 `configs/vlm_lora_multitask.toml`；日常调整先改配置，临时覆盖再用环境变量。launcher
支持 `MAX_STEPS`、`NAVSIM_TRAIN_LIMIT`、`NAVSIM_VAL_LIMIT`、`VAL_SAMPLES_PER_TASK`、
`TASK_SEQUENCE`、`OUTPUT_DIR` 和 `NO_SAVE_CHECKPOINT=1`。`TASK_SEQUENCE` 主要用于复现/诊断。

## Pilot 与正式训练

smoke 成功后，先跑 100–300 updates 的短 pilot，不要立刻跑完整任务：

```bash
MAX_STEPS=200 OUTPUT_DIR=outputs/vlm-lora-multitask-pilot-200 \
  bash scripts/run_vlm_lora_multitask_train.sh
```

当前默认是 batch size 1、梯度累积 1、bf16、SDPA、LoRA rank 8。DriveLM/A-OKVQA 每张图像
默认限制为 230,400 像素，以控制六相机问答的显存；NAVSIM 图像分辨率保持原设置。NAVSIM
轨迹样本在更新之间按配置比例抽样。`metrics.jsonl` 记录每次评估的三个任务训练/验证 loss。观察 loss 是否有限、
各任务是否都在下降后，再生成固定 NAVSIM validation predictions 并计算 ADE/FDE；DriveLM 和
A-OKVQA 还需分别跑答案准确率评估。loss 下降只能证明优化目标在拟合，不能代替这些任务指标。

## 因果隔离实验：只更新 NAVSIM VLM LoRA

为了区分“VLM cache 变化”与“QA 任务干扰”，可以固定已经训练好的 Planning Expert，只让
NAVSIM loss 更新 VLM LoRA，并关闭两个 QA 任务：

```bash
CONFIG_FILE=configs/vlm_lora_navsim_only.toml \
OUTPUT_DIR=outputs/vlm-lora-navsim-only-pilot-200 \
bash scripts/run_vlm_lora_multitask_train.sh
```

配置中的 `train_planner = false` 会冻结 Planning Expert，`task_weights.drivelm = 0` 和
`task_weights.aokvqa = 0` 也会跳过 QA 数据加载。checkpoint 包含 LoRA adapter 和固定 expert，
位于 `outputs/vlm-lora-navsim-only-pilot-200/`。

当前 200-step pilot 在 1006 条 NAVSIM validation 上得到：

| Checkpoint | ADE_4s | FDE_4s |
| --- | ---: | ---: |
| planner-SFT | 0.2717 m | 0.5967 m |
| NAVSIM-only VLM LoRA, expert frozen | **0.2687 m** | **0.5950 m** |
| shared NAVSIM + DriveLM + A-OKVQA LoRA | 0.2751 m | 0.5980 m |

这说明仅用 NAVSIM 更新 VLM cache 并没有造成退化，反而有小幅改善；均权 QA 多任务 LoRA
则明显更差。它支持“QA 更新共享 LoRA 会干扰规划条件”的假设，但还不是完全隔离证明，
因为 shared baseline 同时训练了 Planning Expert。下一步应运行“QA-only LoRA + frozen expert”，
再与本实验做同 seed 的 paired comparison。

### QA-only LoRA 对照

第二个隔离实验只采样 DriveLM 和 A-OKVQA，仍然冻结 Planning Expert：

```bash
CONFIG_FILE=configs/vlm_lora_qa_only.toml \
OUTPUT_DIR=outputs/vlm-lora-qa-only-pilot-200 \
bash scripts/run_vlm_lora_multitask_train.sh
```

该实验的 adapter 随后仍用 NAVSIM validation 推理。若它的规划指标相对固定 expert
明显变差，而 NAVSIM-only LoRA 不变差，就能更直接地把退化归因到 QA loss 对共享 VLM
cache 的改变，而不是 Planning Expert 的更新。

本次 200-step 对照已经完成。两种规划推理都使用模型默认固定 noise seed，因此可以做逐
场景配对比较：

| Checkpoint | ADE_4s | FDE_4s | 相比 NAVSIM-only ADE |
| --- | ---: | ---: | ---: |
| NAVSIM-only VLM LoRA, expert frozen | 0.2687 m | 0.5950 m | 基准 |
| QA-only VLM LoRA, expert frozen | 0.2743 m | 0.6025 m | **+2.10%** |
| shared NAVSIM + DriveLM + A-OKVQA LoRA | 0.2751 m | 0.5980 m | +2.38% |

QA-only 相对 planner-SFT 的 ADE 也上升 `0.96%`；相对 NAVSIM-only 的 paired comparison
中，`403` 个场景变好、`602` 个变差、`1` 个持平。最差 12 个场景的绝对纵向误差平均增加
`0.087 m`，横向误差只增加 `0.005 m`。这组结果支持 QA loss 改变共享 VLM cache、进而
影响 Planning Expert 条件的因果解释。完整预测、指标和轨迹图在
`outputs/navsim-val-vlm-lora-qa-only/` 与 `outputs/navsim-vlm-lora-qa-only-review/`。

正式 checkpoint 包含 `lora_adapter/`、Planning Expert 权重 `model.safetensors`、配置和
`trainer_state.pt`。推理时从原始 4B 模型加载 adapter：

```bash
python scripts/run_vqa.py --model Qwen-Drive-1.0-4B \
  --lora-adapter outputs/vlm-lora-multitask-pilot-200/lora_adapter \
  --image path/to/image.jpg --question "What is happening?"

python scripts/run_planning.py --model Qwen-Drive-1.0-4B \
  --planner outputs/vlm-lora-multitask-pilot-200 \
  --lora-adapter outputs/vlm-lora-multitask-pilot-200/lora_adapter \
  --scenes /path/to/navsim_qwen_val.jsonl --image-root /path/to/navsim_trainval_v1.1 \
  --output outputs/lora-navsim-val/predictions.jsonl --mode direct_planning
```

## 已知范围

- 这是第一版 joint multitask trainer，batch size 固定为 1 个场景/问题；gradient accumulation
  用于模拟更大的有效 batch。
- `navsim_loss` 使用可微 VLM KV cache；它不是冻结 VLM 的旧 `prefill_frozen_vlm()` 路径。
- 当前没有 reasoning 文本监督，也没有把离散生成出的 rationale 放进 LoRA 训练图。
- 当前不启用 gradient checkpointing。A800 上三任务 smoke 已通过；短 smoke 不代表长 pilot
  已完成，正式跑时仍要观察显存和各任务指标。
- 本地 ADE/FDE 是 open-loop 诊断，不是官方 NAVSIM PDM/PDMS；后者还需接通官方 `navtest`
  场景与 metric cache。
- 如果 task val loss 正常但任务指标恶化，应先查学习率、采样比例、任务遗忘和逐场景回归，再
  决定是否加长训练。
