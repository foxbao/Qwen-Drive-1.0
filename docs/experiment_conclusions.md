# 实验结论与后续路线

本文记录截至当前的本地实验结论。除特别说明外，NAVSIM 指本项目整理后的
`navsim_qwen_val.jsonl` 1006 个验证场景，指标是 open-loop ADE/FDE，不等价于官方
NAVSIM PDM/PDMS 闭环评测。

## 1. 已完成的主要结果

| 实验 | ADE (m) | FDE (m) | 说明 |
| --- | ---: | ---: | --- |
| 原始 `planner-sft` | 0.2717 | 0.5967 | 本地 baseline |
| NAVSIM Planning Expert full fine-tune | **0.2580** | **0.5708** | 完整 NAVSIM train split，1 epoch |
| NAVSIM-only VLM LoRA，expert frozen | 0.2687 | 0.5950 | 200-step pilot |
| QA-only VLM LoRA，expert frozen | 0.2743 | 0.6025 | 200-step pilot |
| NAVSIM + DriveLM + A-OKVQA shared LoRA | 0.2751 | 0.5980 | 200-step pilot，expert 也更新 |
| Stage 2 adapter + expert，100-scene pilot | 0.2808 | 0.6177 | 不作为正式 checkpoint |

NAVSIM-only LoRA 与完整 planner 微调的方向一致，说明规划数据更新 VLM 条件并不会天然
破坏规划。QA-only LoRA 在冻结 Planning Expert 时使 ADE 相对 NAVSIM-only 增加约 2.10%；
逐场景比较中 602 个场景变差、403 个变好、1 个持平。最差场景的误差主要沿纵向增长，
横向变化很小。

这组 paired 对照支持以下机制解释：DriveLM/A-OKVQA 的 QA loss 更新了共享 VLM LoRA，
从而改变 NAVSIM 场景的 VLM KV cache；固定的 Planning Expert 接收到不同条件后，规划
轨迹发生退化。它是有控制实验支持的因果线索，但仍不是官方闭环证明。

## 2. Alpamayo-style Stage 2 结论

当前已经验证了“加载 NAVSIM-only adapter、冻结 VLM 和 adapter、只训练 Planning Expert”
的代码机制。100 场景 pilot 的验证结果反而变差，原因更可能是小样本 expert 更新导致的
过拟合或学习率不合适，而不是 adapter 加载本身：

- 用完整 NAVSIM expert 与同一个 NAVSIM-only adapter 直接推理：ADE `0.2589`、FDE `0.5732`，
  与 base VLM 组合的 `0.2580/0.5708` 接近；
- 因此 100-scene Stage 2 checkpoint 不用于正式部署。

正式 Stage 2 应使用完整 NAVSIM train split，降低学习率，保留独立 validation，并在训练
完成后跑完整 1006 场景评估。当前实现仍是本地 open-loop 训练/评估，不声称复现官方
Alpamayo 或 NAVSIM 闭环配方。

## 3. 推荐的模型组织

规划路径推荐使用：

1. NAVSIM-only VLM LoRA adapter；
2. 完整 NAVSIM train split 训练得到的 Planning Expert；
3. direct 或 reasoning-conditioned 模式分别保存和评估，不能混用 cache。

QA 路径使用独立的 QA LoRA。不要把 QA-only adapter 合并进规划部署，也不建议继续扩大
当前均权 shared LoRA，除非先降低 QA 采样/损失权重并重新做 NAVSIM 与 VQA 的 paired 评估。

## 4. 当前条件是否足够

足够进行下一阶段的正式工程实验：

- GPU：NVIDIA A800-SXM4 80GB，当前空闲；
- 数据：NAVSIM train/val、DriveLM、A-OKVQA 均已整理，NAVSIM 为 9018/1006 scenes；
- 磁盘：`/cloud/cloud-ssd1` 约剩 681GB，系统盘约剩 287GB；
- 代码：训练、LoRA、配置化 launcher、open-loop 评估和回归可视化脚本已具备；
- 验证：Python 编译、LoRA 单元测试和已有 smoke test 已通过。

仍然缺少或尚未完成的是官方闭环评测条件：navtest 官方数据/地图、nuPlan/NAVSIM 依赖和
metric cache 尚未接通。因此现在可以继续做完整 Stage 2、独立 QA LoRA 和多任务权重实验，
但结果应标注为本地 open-loop 研究结果；正式 benchmark 结论要等官方 navtest/PDM 流程。

## 5. 建议执行顺序

1. 以完整 NAVSIM split 做低学习率 Stage 2，保存独立 checkpoint。
2. 在同一 1006 场景上评估 base、NAVSIM-only adapter 和 Stage 2 三组结果。
3. 对 Stage 2 做逐场景回归检查，确认是否仍是纵向过拟合。
4. 继续完成 DriveLM held-out VQA 评估，并记录 QA adapter 的能力变化。
5. 下载并接通官方 navtest 后，再报告 PDM/PDMS 或闭环结果。
