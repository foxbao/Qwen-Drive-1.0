# NAVSIM trainval 数据处理说明

本文记录当前机器上 NAVSIM/OpenScene v1.1 数据如何整理、索引并转换成
Qwen-Drive 可读取的 planning scene JSONL。本文描述的是已经跑通的第一版工程流程，
适合做数据读取验证和 Planning Expert 初步微调。

## 1. 数据位置与目录

数据实际位于 `/cloud/cloud-ssd1`，用户可通过下面的路径访问：

```text
/home/baojiali/Downloads/qwen-drive/datasets/navsim_trainval_v1.1/
```

目录结构：

```text
navsim_trainval_v1.1/
├── archives/       # 原始 .tgz 压缩包，作为备份保留
├── sensor_shards/  # 解压后的 navtrain_current/history 分片
├── metadata/       # openscene-v1.1/meta_datas/trainval/*.pkl
├── indexes/        # log 索引、metadata 扫描报告
├── prepared/       # 转换后的 scene JSONL 和 smoke 输出
└── logs/           # 后续运行日志
```

整理时只移动了同一块 SSD 上的顶层目录项，没有重新复制图片；原始压缩包没有被删除或
修改。

## 2. 原始数据各部分的作用

### 2.1 `archives/`

包括：

- `navtrain_current_1.tgz` ～ `navtrain_current_32.tgz`
- `navtrain_history_1.tgz` ～ `navtrain_history_32.tgz`
- `openscene_metadata_trainval.tgz`

这是原始下载备份。后续程序读取解压后的文件，不直接读取压缩包。

### 2.2 `sensor_shards/`

每个分片包含若干 log 目录，例如：

```text
sensor_shards/navtrain_current_12/<log_name>/CAM_F0/*.jpg
sensor_shards/navtrain_history_20/<log_name>/CAM_F0/*.jpg
```

每个 log 中有多路相机和 `MergedPointCloud`。当前 Qwen-Drive camera planner 只使用：

```text
CAM_F0   # front
CAM_L0   # front-left
CAM_R0   # front-right
```

其他相机和 LiDAR 暂时保留，但没有写入 planning scene。

`current` 和 `history` 中的 log 名称会重复，它们提供的是同一 log 的不同时间帧，不能
简单把一个目录复制到另一个目录覆盖掉。

### 2.3 `metadata/openscene-v1.1/meta_datas/trainval/`

每个 `.pkl` 是一个 metadata 列表。单条 metadata 主要包含：

- `token`、`scene_token`、`scene_name`、`log_name`
- `timestamp`、`sample_prev`、`sample_next`
- `cams`：各相机的图片相对路径和标定信息
- `ego2global_translation`、`ego2global_rotation`
- `ego_dynamic_state`：当前速度和加速度
- `driving_command`
- `roadblock_ids`、地图位置和其他 NAVSIM 标注

metadata 是完整 trainval 元数据，而 sensor tarball 只包含 navtrain 子集，因此不是每一条
metadata 都有对应的图片。

## 3. 已执行的处理步骤

### 3.1 建立 sensor/log 索引

脚本：

```text
scripts/index_navsim_shards.py
```

执行命令：

```bash
conda activate qwen-drive
cd /home/baojiali/Downloads/public_code/Qwen-Drive-1.0

python scripts/index_navsim_shards.py \
  --data-root /home/baojiali/Downloads/qwen-drive/datasets/navsim_trainval_v1.1 \
  --scan-metadata
```

生成：

```text
indexes/sensor_log_index.json
indexes/layout_report.json
indexes/metadata_manifest.jsonl
```

这次扫描结果：

```text
current 分片：32
history 分片：32
log 数量：1192
metadata 文件：1310
metadata 记录：723019
完整 current 三路相机记录：103288
完整 history 三路相机记录：49207
current/history 任一来源均有三路相机记录：152495
```

### 3.2 生成 Qwen-Drive scene JSONL

脚本：

```text
scripts/build_navsim_qwen_scenes.py
```

每个候选窗口使用 NAVSIM 风格的：

- 4 帧历史：`-1.5s`、`-1.0s`、`-0.5s`、当前帧
- 10 帧未来 metadata：`0.5s` 到 `5.0s`
- 当前帧使用 `current` 图片
- 3 帧历史使用 `history` 图片
- 窗口内的记录必须属于同一个 `scene_token`
- 默认要求当前帧有 `roadblock_ids`
- 默认跳过 `unknown` driving command

命令：

```bash
python scripts/build_navsim_qwen_scenes.py \
  --data-root /home/baojiali/Downloads/qwen-drive/datasets/navsim_trainval_v1.1 \
  --output /home/baojiali/Downloads/qwen-drive/datasets/navsim_trainval_v1.1/prepared/navsim_qwen_train.jsonl
```

本次生成：

```text
navsim_qwen_train.jsonl：10024 条
文件大小：约 85 MB
```

脚本只写图片相对路径，不复制图片。例如：

```text
sensor_shards/navtrain_history_1/<log>/CAM_F0/<token>.jpg
```

因此读取时的 `--image-root` 应该是：

```text
/home/baojiali/Downloads/qwen-drive/datasets/navsim_trainval_v1.1
```

### 3.3 轨迹转换

当前第一版转换做了以下处理：

1. 从 `ego2global_translation` 和 `ego2global_rotation` 得到全局自车位姿；
2. 把历史和未来位姿转换到当前帧自车坐标系；
3. 将 metadata 的 2 Hz pose 线性重采样到 10 Hz；
4. 将速度和加速度向量旋转到当前帧坐标系后重采样；
5. 生成 Qwen-Drive 需要的 `[16, 3]` 历史轨迹和 `[50, 3]` 未来轨迹；
6. 将 NAVSIM driving command 映射为 Qwen-Drive 的：
   - `0`: straight
   - `1`: left
   - `2`: right

当前转换生成的核心字段为：

```text
hist_traj_10hz          [16, 3]
hist_vel_10hz           [16, 2]
hist_acc_10hz           [16, 2]
future_traj_10hz        [50, 3]
future_valid_mask_10hz  [50]
ego_status
nav_command
```

### 3.4 训练/验证划分

脚本：

```text
scripts/split_scene_file.py
```

命令：

```bash
python scripts/split_scene_file.py \
  --input /home/baojiali/Downloads/qwen-drive/datasets/navsim_trainval_v1.1/prepared/navsim_qwen_train.jsonl \
  --train-output /home/baojiali/Downloads/qwen-drive/datasets/navsim_trainval_v1.1/prepared/navsim_qwen_train_split.jsonl \
  --val-output /home/baojiali/Downloads/qwen-drive/datasets/navsim_trainval_v1.1/prepared/navsim_qwen_val.jsonl \
  --val-fraction 0.1 \
  --seed 3407
```

本次结果：

```text
训练集：9018 条
验证集：1006 条
训练 scene_token：6954 个
验证 scene_token：786 个
train/val scene_token 重叠：0
```

按 `scene_token` 而不是按单条窗口随机划分，是为了避免同一场景的相邻窗口同时进入
训练集和验证集。

## 4. 输出文件内容

### `sensor_log_index.json`

建立：

```text
log_name -> current/history 对应的实际分片目录
```

它不包含图片数据，只是路径索引。

### `layout_report.json`

记录分片数量、log 数量、缺失分片以及 metadata 扫描汇总。

### `metadata_manifest.jsonl`

每行对应一个 `.pkl` 文件，记录：

- metadata 文件相对路径
- 文件大小
- metadata 记录数
- 可找到完整 current 三路相机的记录数
- 可找到完整 history 三路相机的记录数
- 任一 current/history 来源可找到三路相机的记录数
- metadata 中找不到 sensor log 的记录数

### `navsim_qwen_train.jsonl`

未划分的全部转换结果，每行一个 Qwen-Drive planning scene。主要结构：

```json
{
  "messages": {"...": "ChatML 用户图片和指令"},
  "meta_info": {"...": "dataset/token/scene_token/cam_order"},
  "trajectory": {
    "hist_traj_10hz": "[16, 3]",
    "hist_vel_10hz": "[16, 2]",
    "hist_acc_10hz": "[16, 2]",
    "future_traj_10hz": "[50, 3]",
    "future_valid_mask_10hz": "[50]",
    "ego_status": "速度、加速度、驾驶指令",
    "nav_command": "0/1/2"
  }
}
```

### `navsim_qwen_train_split.jsonl`

正式用于 Planning Expert 微调的训练 scene 文件。

### `navsim_qwen_val.jsonl`

用于训练期间计算验证 loss 的 scene 文件。

### `navsim_qwen_smoke.jsonl`

只有 10 条样本，用于验证数据读取和模型前向，不用于正式训练。

### `navsim_smoke_demo.png`

用真实 Qwen-Drive 模型对转换后的 NAVSIM 样本运行 planning 后生成的可视化结果。

## 5. 已完成的验证

已经验证：

- JSONL 可以被 `read_scene_file` 读取；
- 每条样本包含 12 张图片；
- 所有图片路径存在且可以用 PIL 打开；
- 历史轨迹 shape 为 `[16, 3]`；
- 未来轨迹 shape 为 `[50, 3]`；
- train/val 没有 scene token 泄漏；
- 真实 Qwen-Drive VQA、direct planning、reasoning planning 均可运行。

## 6. “严格 NAVSIM 复现/正式评测”是什么意思

当前转换的目标是：让 NAVSIM 数据进入 Qwen-Drive 的监督训练接口。它已经满足模型输入
格式，但轨迹处理仍是本项目的工程版：使用 OpenScene metadata 的 2 Hz 位姿，再线性
重采样到 10 Hz。

严格 NAVSIM 复现指的是：生成与 NAVSIM v1.1 官方数据加载和评测完全一致的轨迹和场景，
包括：

- 官方 scene window 和 rear-axle 坐标约定；
- 官方 nuPlan/NAVSIM 的位姿、速度、加速度读取方式；
- 官方 2 Hz 场景帧到 10 Hz 评测轨迹的插值方式；
- 官方异常轨迹过滤和有效帧规则；
- NAVSIM PDM 评测所需的地图和 metric cache。

NAVSIM v1.1 的场景数据按 `0.5s` 数据库帧组织，PDM 评测使用 4 秒、10 Hz 的轨迹；
官方代码会在评测路径中进行相应采样/插值。参考
[NAVSIM v1.1 dataclasses.py](https://github.com/autonomousvision/navsim/blob/v1.1/navsim/common/dataclasses.py)
和
[NAVSIM v1.1 navsim_scenario.py](https://github.com/autonomousvision/navsim/blob/v1.1/navsim/planning/scenario_builder/navsim_scenario.py)。

Qwen-Drive 训练接口则要求 5 秒、50 点的 10 Hz future trajectory。因此两者并不完全是
同一个时间网格：

- 当前 JSONL：为 Qwen-Drive 训练准备的 5 秒/50 点轨迹；
- NAVSIM PDM 评测：通常使用前 4 秒，并按照官方评测方式插值。

这意味着：

- 当前数据可以用于读取验证、模型前向和初步 Planning Expert SFT；
- 如果要把最终指标与 NAVSIM leaderboard 或官方基线直接比较，就需要再接入 NAVSIM
  v1.1/nuPlan 官方轨迹处理和评测链路；
- 这不是重新下载图片，而是替换/校准轨迹生成和评测部分。

## 7. Planning Expert 训练 smoke test

已经使用训练集前 10 条 NAVSIM scene 做了真实训练 smoke test。命令为：

```bash
conda activate qwen-drive
cd /home/baojiali/Downloads/public_code/Qwen-Drive-1.0

conda run --no-capture-output -n qwen-drive \
  python scripts/train_planner_real_smoke.py \
  --model /home/baojiali/Downloads/public_code/Qwen-Drive-1.0/Qwen-Drive-1.0-4B \
  --planner /home/baojiali/Downloads/public_code/Qwen-Drive-1.0/Qwen-Drive-1.0-4B/planner-sft \
  --scenes /home/baojiali/Downloads/qwen-drive/datasets/navsim_trainval_v1.1/prepared/navsim_qwen_train_split.jsonl \
  --image-root /home/baojiali/Downloads/qwen-drive/datasets/navsim_trainval_v1.1 \
  --limit 10 \
  --steps 5 \
  --learning-rate 1e-5 \
  --seed 3407 \
  --dtype bfloat16 \
  --attn-implementation sdpa \
  --output /home/baojiali/Downloads/public_code/Qwen-Drive-1.0/outputs/planner-navsim-smoke-10
```

结果：

```text
loss: 0.000012 -> 0.000003
Planning Expert 梯度：正常
冻结 VLM：正常
checkpoint 保存：正常
checkpoint 重新加载：正常
```

输出目录：

```text
outputs/planner-navsim-smoke-10/
├── model.safetensors
└── config.json
```

这个 checkpoint 只用于验证训练链路，不代表最终模型质量。下一步可以使用完整的
`navsim_qwen_train_split.jsonl`（9018 条）进行正式的 Planning Expert 微调；完整训练
预计会明显长于 smoke test，建议先固定训练参数、日志目录和 checkpoint 保存策略。

## 8. 手动启动正式训练

训练参数已经从启动脚本中抽离到 TOML 配置文件：

```text
configs/navsim_planner_train.toml
```

配置文件包含模型路径、数据路径、输出目录、epoch、学习率、设备、dtype、attention
backend 等参数。正式 trainer 会解析该文件，命令行参数优先级高于配置文件。

已经提供 NAVSIM 专用启动脚本：

```text
scripts/run_navsim_planner_train.sh
```

查看参数说明：

```bash
bash scripts/run_navsim_planner_train.sh --help
```

也可以直接调用 Python trainer：

```bash
conda run --no-capture-output -n qwen-drive \
  python scripts/train_planner.py \
  --config configs/navsim_planner_train.toml
```

先跑 100 条样本：

```bash
LIMIT=100 VAL_LIMIT=100 \
  bash scripts/run_navsim_planner_train.sh
```

跑完整训练集：

```bash
bash scripts/run_navsim_planner_train.sh
```

默认设置为 `cuda:0`、`bfloat16`、`sdpa`、1 个 epoch，并从原始
`Qwen-Drive-1.0-4B/planner-sft` 开始。训练输出默认写入：

```text
outputs/planner-navsim-epoch1/
```

正式 trainer 会显示训练和验证的 tqdm 进度条，包含已处理 scene 数量、当前 loss 和
optimizer step；旧版本只在整个 epoch 结束后输出一次，容易看起来像卡住。

当前 trainer 的实际 `batch_size` 固定为 1；`--batch-size 2` 会被脚本拒绝。可以通过：

```bash
GRAD_ACCUM_STEPS=4 bash scripts/run_navsim_planner_train.sh
```

让 4 个单样本的梯度累积后再更新一次参数，相当于有效 batch size 为 4，但不会把 4 个
完整 VLM cache 同时放进显存。这通常比直接增大 batch 更稳妥，但不会使每个样本的视觉
prefill 变成 4 倍快。真正的 `batch_size > 1` 需要额外实现输入 padding、变长图像 token
和 cache batching，不能只修改一个命令行参数。

如果训练中断并且已有完整 epoch 的 checkpoint，可以这样继续：

```bash
RESUME=outputs/planner-navsim-epoch1 \
  bash scripts/run_navsim_planner_train.sh
```

如果要做另一组实验，复制配置文件后修改副本即可：

```bash
cp configs/navsim_planner_train.toml configs/navsim_planner_lr2e-5.toml
sed -i 's/learning_rate = 1e-5/learning_rate = 2e-5/' \
  configs/navsim_planner_lr2e-5.toml
CONFIG_FILE=configs/navsim_planner_lr2e-5.toml \
  bash scripts/run_navsim_planner_train.sh
```
