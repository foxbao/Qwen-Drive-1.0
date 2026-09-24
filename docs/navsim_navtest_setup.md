# NAVSIM v1.1 `navtest` 数据准备

## 最简单的下载方式

`navtest` 不是单独一个文件。官方 v1.1 流程需要地图和 OpenScene test 数据；评测 cache
是数据下载后再本地生成的。已经下载好的 NAVSIM trainval 不用重下。

确认 `/cloud/cloud-ssd1` 有足够空间后，在服务器终端依次运行：

```bash
mkdir -p /cloud/cloud-ssd1/navsim_workspace
git clone --branch v1.1 --single-branch \
  https://github.com/autonomousvision/navsim.git \
  /cloud/cloud-ssd1/navsim_workspace/navsim
cd /cloud/cloud-ssd1/navsim_workspace/navsim/download
bash download_maps.sh
bash download_test.sh
```

两条脚本会从官方源下载并解压；地图脚本处理 map 包，test 脚本处理 test metadata、32 个
camera shards 和 32 个 LiDAR shards。不需要手动从 ModelScope 页面逐个点 shard，也不要再跑
trainval 下载脚本。请先遵守 NAVSIM/nuPlan 和 OpenScene 的数据许可。

检查文件与剩余空间：

```bash
df -h /cloud/cloud-ssd1
du -sh /cloud/cloud-ssd1/navsim_workspace/navsim/download
```

下载会持续一段时间；完成后先停在这里即可。此时文件已下载，但还没有生成 `navtest` metric
cache，也还不能直接用 Qwen-Drive 的本地 JSONL 推理脚本提交官方评测。

## 之后的官方 cache 步骤

后续需要设置 NAVSIM 的 `NAVSIM_DEVKIT_ROOT`、`OPENSCENE_DATA_ROOT`、`NUPLAN_MAPS_ROOT` 和
`NAVSIM_EXP_ROOT`，并在 NAVSIM v1.1 环境下运行官方
`scripts/evaluation/run_metric_caching.sh`（split 为 `navtest`）。我们可以等下载完后根据实际
目录结构一起设置这些变量，避免手动挪错大文件。

metric cache 完成后，还需给 Qwen-Drive 加一个官方 NAVSIM scene/提交 adapter：映射相机路径、
坐标、预测时间网格和提交格式，再调用官方 PDM scorer。当前本地 ADE/FDE 仅为 open-loop
诊断，不能称为官方 PDMS。

## 官方参考

- [NAVSIM v1.1 安装与下载说明](https://github.com/autonomousvision/navsim/blob/v1.1/docs/install.md)
- [NAVSIM v1.1 地图下载脚本](https://github.com/autonomousvision/navsim/blob/v1.1/download/download_maps.sh)
- [NAVSIM v1.1 test 下载脚本](https://github.com/autonomousvision/navsim/blob/v1.1/download/download_test.sh)
- [NAVSIM v1.1 metric caching 脚本](https://github.com/autonomousvision/navsim/blob/v1.1/scripts/evaluation/run_metric_caching.sh)
- [OpenScene 数据获取说明](https://github.com/OpenDriveLab/OpenScene/blob/main/docs/getting_started.md)
