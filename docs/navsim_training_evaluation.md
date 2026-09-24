# NAVSIM Planning 验证与指标计算

本文记录如何用 NAVSIM 验证集生成 Planning Expert 预测，并计算开环位移指标。当前
流程评估的是本项目转换后的 NAVSIM trainval validation split，不是官方 navtest 闭环
PDMS 评测。官方 navtest 的地图、test shards 和 metric-cache 准备说明见
[`docs/navsim_navtest_setup.md`](navsim_navtest_setup.md)。

## 1. 生成验证集预测

在项目根目录运行。下面以训练完成的 checkpoint 为例：

```bash
conda run --no-capture-output -n qwen-drive \
  python scripts/run_planning.py \
  --model Qwen-Drive-1.0-4B \
  --planner outputs/planner-navsim-full-epoch1 \
  --scenes /home/baojiali/Downloads/qwen-drive/datasets/navsim_trainval_v1.1/prepared/navsim_qwen_val.jsonl \
  --image-root /home/baojiali/Downloads/qwen-drive/datasets/navsim_trainval_v1.1 \
  --output outputs/navsim-val-direct/predictions.jsonl \
  --mode direct_planning \
  --num-samples 1 \
  --num-workers 4 \
  --device cuda:0 \
  --dtype bfloat16
```

脚本逐场景生成并写入 JSONL。意外中断后，重新执行同一命令会跳过已完成场景并继续；
如果要做另一组实验，建议使用不同的 `--output` 路径，保留每次实验的结果。

## 2. 计算位移指标

评估参数放在 [`configs/navsim_eval.toml`](../configs/navsim_eval.toml)。默认配置指向上一步
的预测文件和输出目录。运行：

```bash
bash scripts/run_navsim_eval.sh
```

指标 JSON 会写到：

```text
outputs/navsim-val-direct/navsim_metrics.json
```

如需临时评估另一份预测文件，可用环境变量覆盖 TOML：

```bash
PREDICTIONS=outputs/my-run/predictions.jsonl \
OUTPUT_DIR=outputs/my-run \
bash scripts/run_navsim_eval.sh
```

也可以用 `CONFIG_FILE=...` 指向另一份 TOML。直接运行 Python 时，命令行参数同样可以
覆盖配置：

```bash
conda run --no-capture-output -n qwen-drive \
  python scripts/eval_navsim.py \
  --config configs/navsim_eval.toml \
  --predictions outputs/my-run/predictions.jsonl \
  --output outputs/my-run
```

如果配置了官方 NAVSIM metric cache，评估器还会计算 PDM 指标；这需要另行准备官方
NAVSIM 依赖、nuPlan maps 和 metric cache。当前本地 trainval validation 评估没有启用它。

## 3. 指标含义与限制

- `ADE_4s`：预测和真值前 4 秒（40 个 10 Hz 点）二维位置距离的平均值，单位为米。
- `FDE_4s`：第 4 秒末端的二维位置距离，单位为米。
- 这两个位移指标不评估 heading，也不是碰撞、道路合规或闭环仿真分数。
- 本次推理使用 `direct_planning`、每场景一个候选，因此不是 best-of-N/minADE。
- 本地 scene 的未来轨迹由当前数据准备流程从 OpenScene 2 Hz metadata 线性重采样，和
  官方 NAVSIM PDM 输入/仿真评测协议不完全等价。

## 4. 当前训练 checkpoint 的结果

`outputs/planner-navsim-full-epoch1` 与原始 `Qwen-Drive-1.0-4B/planner-sft` 在同一份
1006 条本地验证样本上进行了对照。两次都使用 `direct_planning`、单候选和相同推理配置：

| Checkpoint | ADE_4s | FDE_4s | 样本数 |
| --- | ---: | ---: | ---: |
| 原始 `planner-sft` | 0.2717 m | 0.5967 m | 1006 |
| NAVSIM 微调 1 epoch | **0.2580 m** | **0.5708 m** | 1006 |

微调后 ADE 相对下降约 5.06%，FDE 相对下降约 4.33%。逐场景配对比较中，微调版在
1006 个场景中的 552 个场景 ADE 更低、454 个更高；FDE 在 542 个场景更低、457 个更高，
另有 7 个持平。因此总体改善幅度较小但方向一致，不是每个场景都变好。

这支持“本次 NAVSIM 微调对这份本地 open-loop 验证 split 有小幅正向作用”的结论；
它仍不能证明官方 NAVSIM 闭环 PDMS 会提升。仓库 README 中其他数据集或官方闭环协议的
结果也不能直接与这里横向比较。

### Alpamayo-style adapter Stage 2 pilot

Planning Expert trainer 现在支持在冻结的 VLM LoRA 上训练：

```bash
CONFIG_FILE=configs/navsim_stage2_navsim_adapter_pilot.toml \
bash scripts/run_navsim_planner_train.sh
```

配置中的 `lora_adapter` 指向 NAVSIM-only VLM adapter；Stage 2 只更新 Planning Expert。
本次 100-scene pilot 的 train/validation loss 分别为 `3.7e-5` 和 `2.3e-5`，说明加载和
冻结逻辑正常。但在完整 1006-scene open-loop validation 上得到 `ADE=0.2808 m`、
`FDE=0.6177 m`，差于使用同一 adapter、未继续训练的 `0.2687 m / 0.5950 m`。逐场景
比较中 523 个场景 ADE 变差，最差 12 个场景的绝对纵向误差平均增加 `0.319 m`，横向仅
增加 `0.007 m`。因此这个小 pilot 不能作为正式 checkpoint；正式 Stage 2 应使用完整
训练 split，并重新调低学习率或增加正则化后再评估。

Stage 2 结果和轨迹图位于：

```text
outputs/planner-navsim-stage2-adapter-pilot-100/
outputs/navsim-stage2-adapter-review/stage2-vs-stage1-adapter/
```

作为控制实验，把已经在完整 NAVSIM split 上训练好的 `planner-navsim-full-epoch1` 与
NAVSIM-only adapter 组合，而不再重新训练 expert，得到 `ADE=0.2589 m`、`FDE=0.5732 m`。
原始 full expert（base VLM）是 `0.2580 m / 0.5708 m`，说明 NAVSIM adapter 本身几乎不
破坏规划；Stage 2 pilot 的退化主要来自 100 场景小样本 expert 更新。控制实验的结果和
逐场景图在：

```text
outputs/navsim-val-full-expert-navsim-adapter/
outputs/navsim-full-expert-adapter-review/
```

因此当前推荐的落地方式是：QA 使用独立 adapter，规划部署使用 NAVSIM adapter 与 full-data
Planning Expert；不要把 QA adapter 合并进规划路径，也不要把 100-scene Stage 2 pilot 当作
正式 checkpoint。正式 Alpamayo-style Stage 2 仍应在完整 NAVSIM split 上训练，并先用较低
学习率做一轮完整实验。

### Reasoning planning 对照

之后还用 `reasoning_planning` 模式在相同 1006 条样本上比较了两个 checkpoint：

| Checkpoint | ADE_4s | FDE_4s | 样本数 |
| --- | ---: | ---: | ---: |
| 原始 `planner-sft` | 0.2731 m | 0.6024 m | 1006 |
| NAVSIM 微调 1 epoch | **0.2632 m** | **0.5845 m** | 1006 |

微调版的 ADE 相对下降约 3.60%，FDE 相对下降约 2.96%；逐场景比较中，ADE 在 535 个场景
更低、471 个更高，FDE 在 530 个更低、469 个更高、7 个持平。两边生成的 reasoning 都非空，
平均约 72 个字符。样例文本简洁可读，但这不是正式的语言质量评测。

因此，这轮 direct-only 专家微调在当前本地验证集上没有显示出对 reasoning planning
的退化，反而也有小幅改善。它并不等于已经进行了 reasoning-conditioned SFT：旧训练器
始终用 direct prompt 生成 VLM cache。现在训练入口已经增加 `conditioning_mode =
"reasoning"`，会按推理时相同的方式由冻结 VLM 贪心生成理由，再用其 cache 训练专家。
这建立了训练机制，但不会监督理由文字本身；完整训练前先跑小样本 pilot，确认耗时、生成
文本和验证表现，再决定是否跑一轮正式对照。

### 共享 VLM LoRA pilot 对照

`outputs/vlm-lora-multitask-pilot-200` 是一个 200-update 的共享 LoRA pilot：它在每个 update
中按相同权重抽取 NAVSIM、DriveLM 或 A-OKVQA；NAVSIM 同时更新 LoRA 和 Planning Expert，两个
QA 任务只更新 LoRA。完整 1006 场景 validation 的 direct 结果如下：

| Checkpoint | ADE_4s | FDE_4s | 相比 NAVSIM-only fine-tune 的 ADE |
| --- | ---: | ---: | ---: |
| 原始 `planner-sft` | 0.2717 m | 0.5967 m | +5.33% |
| NAVSIM-only 1 epoch | **0.2580 m** | **0.5708 m** | 基准 |
| 200-step shared VLM LoRA | 0.2751 m | 0.5980 m | +6.63% |

LoRA pilot 比原始 planner-SFT 的 ADE 也高 1.24%。配对比较中它相对 NAVSIM-only fine-tune
在 588/1006 个场景有更高 ADE；最差 12 个回归场景平均绝对纵向误差增加约 0.287 m，横向变化仅
0.010 m。因此当前证据不支持直接延长这一组均权多任务训练。后续应先查看 VQA held-out 指标，
再通过 NAVSIM-only LoRA、VQA-only LoRA 或降低 QA 采样/损失权重来隔离干扰来源。

该对照的 prediction、summary、逐场景 CSV 和 24 张轨迹图在：

```text
outputs/navsim-val-vlm-lora-full/
outputs/navsim-vlm-lora-full-review/lora-vs-planner-sft/
outputs/navsim-vlm-lora-full-review/lora-vs-navsim-full-ft/
```

对应预测和指标文件分别保存在：

```text
outputs/navsim-val-reasoning-sft-baseline/
outputs/navsim-val-reasoning-finetuned/
```

### 逐场景退化检查与轨迹图

已经加入可重复运行的逐 token 配对检查脚本。它会对比上述 direct/reasoning 两组
baseline 与 fine-tuned 预测，输出全场景 CSV/JSON，并分别画出 ADE 增幅最大的 12 个
场景：

```bash
bash scripts/run_navsim_regression_review.sh
```

输出位置：

```text
outputs/navsim-regression-review/direct/summary.json
outputs/navsim-regression-review/direct/per_scene.csv
outputs/navsim-regression-review/direct/plots/
outputs/navsim-regression-review/reasoning/summary.json
outputs/navsim-regression-review/reasoning/per_scene.csv
outputs/navsim-regression-review/reasoning/plots/
```

本轮重新配对 1006/1006 条，结果与上表一致。最大的 direct ADE 退化来自 token
`e5d6e01f41c45df5`：ADE 从 0.158 m 增至 0.461 m，FDE 从 0.628 m 增至 1.432 m；最大
reasoning ADE 退化来自 `06a3c0d706f3593c`：ADE 从 0.230 m 增至 0.513 m，FDE 从 0.421 m
增至 1.067 m。对各自最差的 12 个场景做坐标分量误差拆分后，平均绝对纵向误差增加
0.233 m（direct）/0.221 m（reasoning），平均绝对横向误差变化仅 +0.005 m/-0.000 m；
抽看的轨迹图也显示误差主要是在约 1 秒后沿纵向逐渐拉大。这提示有些退化场景存在向前
走得过多的趋势，但不是场景原因定论。图中没有地图/相机图像，也没有碰撞和
道路合规信息；需结合原始场景画面或 NAVSIM PDM 才能判断是否属于速度、路口/障碍物或
真值重采样问题。

通过 `CONFIG_FILE=...`、`OUTPUT_DIR=...`、`TOP_K=...` 可更换评估配置、输出位置或退化
场景数量。该脚本按 ADE 退化排序，同时在 CSV 中列出每个场景的 ADE/FDE 前后值和变化。
