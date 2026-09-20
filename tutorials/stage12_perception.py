"""Stage 12：感知分支 —— 同一个 VLM，另一个任务头

【定位】这是规划主线之后的**可选架构概览**，不是规划的前置课程；真实 CUDA/BEVFormer
实现请继续阅读 `src/qwen_drive_perception/`。本脚本只做形状追踪和一个玩具体素池化，
不是完整的感知模型复现。

【目的】Qwen-Drive 的第三个能力（除了 VQA 和规划）是 **BEV 三维感知**：
3D 检测 + 语义占据 + BEV 地图分割。它和规划**完全并列**，不是规划的前置模块。

【最重要的一句话】感知头**没有自己的 backbone**。它读的是 VLM 已经算出来的特征：

    ┌─ 规划：VLM 的 attention cache ─► Planning Expert ─► 轨迹
    └─ 感知：VLM 的 hidden states   ─► BEV 头 ─────────► 3D 框 / 占据 / 地图

两条支路在 VLM 之后**分叉**，各自有自己的参数，各自独立训练。README 里的说法是
感知头「serves as a probe of the 3D information accessible from the shared
representations」—— 它更像一个**探针**，用来检验共享表征里到底有多少三维信息。

【两个特征抽头（这是感知的核心设计）】

  ┌ LLM 抽头（主路）
  │   最后一层 decoder 的 hidden states，在**图像 token 的位置**上取出
  │   → 再过 `language_model.norm`（因为头是在 post-norm 上训的）
  │   形状 [相机数, 16, 28, 2560]
  │
  └ ViT 抽头（UVTR 路）
      在 `visual.merger` 上挂 forward pre-hook，抓 **merge 之前**的 patch 特征
      → 再补一次 `merger.norm`
      形状 [相机数, 32, 56, 1024]

【坐标约定】
    ego 坐标系：x 前、y 左、z 上（和规划那边一致，但这里多了 z）
    检测 BEV  ：200×200 格，每格 0.512 m → 102.4 m × 102.4 m，z 范围 [-5.0, 5.4]
    占据      ：200×200×16，nuScenes [-40,-40,-1, 40,40,5.4] / nuPlan [-50,-50,-4, 50,50,4]
    地图      ：200×400 格，每格 0.15 m → 60 m（前）× 30 m（侧）
    检测输出  ：[x, y, z, w, l, h, yaw, vx, vy]，yaw 绕 +Z，w 沿车头方向
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def banner(title: str) -> None:
    print("\n" + "=" * 66)
    print(title)
    print("=" * 66)


def toy_voxel_pool(depth, features, voxel_coords, voxel_shape):
    """`voxel_pool_depth` 的纯 Python 版：把视锥的特征按深度权重散射进体素。

    真实实现是 CUDA kernel（`ops/voxel_pool/`），一次融合「深度加权 + 散射求和」，
    中间**不落地** [B, N, C, D, H, W] 这个大张量。

        out[b, cam, x, y, z, c] = Σ feats[b, cam, c, h, w] · depth[b, cam, d, h, w]
                                   （对所有落在同一体素的 (d, h, w) 求和）
    """
    bx, by, bz = voxel_shape
    channels = features.shape[2]          # features 是 (B, N_cam, C, H, W)
    out = torch.zeros(voxel_coords.shape[0], voxel_coords.shape[1], bx, by, bz, channels)
    d, h, w = depth.shape[-3:]
    for b in range(depth.shape[0]):
        for cam in range(depth.shape[1]):
            for di in range(d):
                for hi in range(h):
                    for wi in range(w):
                        x, y, z = voxel_coords[b, cam, 0, di, hi, wi].tolist()
                        if 0 <= x < bx and 0 <= y < by and 0 <= z < bz:
                            out[b, cam, x, y, z] += (
                                features[b, cam, :, hi, wi] * depth[b, cam, di, hi, wi]
                            )
    return out


def main() -> None:
    banner("① 感知头读什么：两个抽头，同一个 VLM")
    num_cams = 6
    llm_grid = (16, 28)
    vit_grid = (32, 56)
    print(f"   {'抽头':<10} {'来源':<34} {'形状':<26} {'通道'}")
    print(f"   {'-' * 78}")
    print(f"   {'LLM':<10} {'decoder 最后一层 hidden_states':<34} "
          f"{f'[{num_cams}, {llm_grid[0]}, {llm_grid[1]}]':<26} 2560")
    print(f"   {'ViT':<10} {'visual.merger 的**输入**（merge 前）':<34} "
          f"{f'[{num_cams}, {vit_grid[0]}, {vit_grid[1]}]':<26} 1024")
    print(f"\n   图像输入是 896×512（patch 16，merge 2）：")
    print(f"     ViT patch 网格   {vit_grid[0]}×{vit_grid[1]}  = 512/16 × 896/16  ✓")
    print(f"     merge 之后       {llm_grid[0]}×{llm_grid[1]}  = {vit_grid[0]}//2 × {vit_grid[1]}//2  ✓")
    print(f"   所以两个抽头是**同一张图的两种分辨率**，不是两份不同的输入。")

    print(f"\n   为什么抽 merge 之前的 ViT 特征？")
    print(f"     主路（LLM 抽头）的网格太粗（16×28），做稠密 BEV 预测不够；")
    print(f"     UVTR 路需要更细的空间网格来投影深度（32×56），所以要用 merge 前的 patch。")
    print(f"   两个抽头**各补一次 norm**，因为 checkpoint 里的特征是 post-norm 的。")

    banner("② 两条支路各自的第一个模块：都是 SimpleFPN")
    print(f"   {'支路':<10} {'输入':<16} {'scale_factors':<20} {'输出层级'}")
    print(f"   {'-' * 66}")
    print(f"   {'主路':<10} {'2560 通道':<16} {'(4.0, 2.0, 1.0, 0.5)':<20} "
          f"{'4 级金字塔，各 256 通道'}")
    print(f"   {'UVTR':<10} {'1024 通道':<16} {'(1.0,)':<20} {'1 级，256 通道'}")
    print(f"\n   主路的四级金字塔喂给空间交叉注意力（多尺度可变形注意力）；")
    print(f"   UVTR 的单级只喂给 DepthNet。")
    print(f"   四级的分辨率（toy 尺寸）：")
    print(f"     {'scale':>6} {'网格':>12}   说明")
    for scale, grid, note in (
        (4.0, "64×112", "两次 ConvTranspose2d 上采样"),
        (2.0, "32×56", "一次上采样"),
        (1.0, "16×28", "恒等（原分辨率）"),
        (0.5, "8×14", "MaxPool2d 降采样"),
    ):
        print(f"     {scale:>6} {grid:>12}   {note}")

    banner("③ UVTR 路：视锥 → 深度 → 体素池化（感知分支主要依赖 CUDA ops 的地方）")
    print(f"   步骤：")
    print(f"     1. DepthNet 从 ViT 特征预测深度分布")
    print(f"        输入  [B*N, 256, 32, 56]")
    print(f"        输出  [B*N, 118, 32, 56] → softmax(dim=1)")
    print(f"        118 = (60.0 - 1.0) / 0.5   ← 深度 1~60 m，每 0.5 m 一格")
    print(f"     2. 预计算视锥坐标（frustum grid）")
    print(f"        网格 56×32×118（W×H×D），像素格 16 px")
    print(f"        把 (u, v) 乘上 z 变成齐次射线 (u·z, v·z, z, 1)，再用 inv(lidar2img) 投回 3D")
    print(f"     3. 体素池化（CUDA op `voxel_pool_depth`）")
    print(f"        对每个体素，把落进来的所有视锥格子的「特征 × 深度概率」求和")
    print(f"        输出 [B, N_cam, 200, 200, 16, 256]（fp32）")
    print(f"     4. 沿相机维求和 → 3 层 Conv3d + BN3d + ReLU")
    print(f"        输出 [B, 256, 16, 200, 200]")

    print(f"\n   ── 跑一个玩具版，把「深度加权散射」这一步看清楚 ──")
    torch.manual_seed(0)
    b, n, d, h, w, c = 1, 2, 4, 2, 2, 3
    depth = torch.rand(b, n, d, h, w)
    depth = depth / depth.sum(dim=2, keepdim=True)          # 沿深度维归一化（softmax 的效果）
    features = torch.randn(b, n, c, h, w)
    voxel_shape = (3, 3, 2)
    # 给每个视锥格子随机指派一个体素坐标
    coords = torch.randint(0, 3, (b, n, 1, d, h, w, 3))
    pooled = toy_voxel_pool(depth, features, coords, voxel_shape)
    print(f"     深度分布      {tuple(depth.shape)}   沿 dim=2 求和 = {depth.sum(dim=2).flatten()[:3].tolist()}")
    print(f"     特征          {tuple(features.shape)}")
    print(f"     体素坐标      {tuple(coords.shape)}   （最后一维是 xyz）")
    print(f"     池化结果      {tuple(pooled.shape)}")
    nonzero = int((pooled.abs().sum(dim=-1) > 0).sum())
    print(f"     非空体素      {nonzero} / {pooled.shape[0] * pooled.shape[1] * voxel_shape[0] * voxel_shape[1] * voxel_shape[2]}")
    print(f"\n   真实规模下（6 相机、118 深度格、32×56、200×200×16 体素），")
    print(f"   中间张量 [1, 6, 256, 118, 32, 56] ≈ 3.2 亿个数（约 1.3 GB）。")
    print(f"   **CUDA kernel 的价值就在于从不把那个张量落下来** —— 直接在寄存器里累加。")

    banner("④ 六层 BEV 编码器：BEVFormer 那一套")
    print(f"   BEV query 的初始化（这一行很关键）：")
    print(f"     bev_queries = self.bev_embedding.weight + uvtr_bev_feat")
    print(f"   —— 学出来的 200×200 位置 embedding，**加上** UVTR 投出来的 BEV 特征。")
    print(f"   是「加」不是「拼」也不是 cross-attention：UVTR 在这里起的是")
    print(f"   **几何先验**的作用，负责把 query 初始化到一个合理的起点。")
    print(f"\n   然后 6 层，每层：")
    print(f"     ① 时域自注意力（deformable self-attention over BEV）")
    print(f"     ② 空间交叉注意力（读 4 级图像金字塔，deformable）")
    print(f"     ③ FFN")
    print(f"\n   ⚠️ 时域那块**在单帧推理下会退化**：")
    print(f"     代码里 shift=[0,0]、prev_bev=None，BEV 队列被构造成 [query, query]，")
    print(f"     自注意力退化成普通的 BEV 自注意力，输出再对「假的队列」取平均。")
    print(f"     队列的机器（num_bev_queue=2 等）留着是为了**兼容 checkpoint 的键名**。")

    banner("⑤ 三个头：各自输出什么")
    print(f"   {'头':<14} {'查询/输入':<26} {'输出形状':<22} {'类别数'}")
    print(f"   {'-' * 76}")
    print(f"   {'3D 检测':<14} {'900 个 object query':<26} {'top-300 个框':<22} {'7'}")
    print(f"   {'占据':<14} {'BEV 特征裁到 occ 范围':<26} {'[200, 200, 16]':<22} {'10'}")
    print(f"   {'地图':<14} {'BEV 特征 grid_sample':<26} {'[200, 400]':<22} {'6'}")
    print(f"\n   检测的 7 类：vehicle, czone_sign, bicycle, generic_object, pedestrian,")
    print(f"               traffic_cone, barrier")
    print(f"   占据的 10 类：上面 7 类 + driveable, background, empty")
    print(f"   地图的 6 类：background, driveable_surface, road_line, road_edge,")
    print(f"               crosswalk, walkway")

    print(f"\n   **检测是无 NMS 的**（NMS-free）：900 个 query × 7 类的 sigmoid 分数")
    print(f"   摊平成 6300 个候选，取全局 top-300，再用中心点范围 [-61.2, 61.2] 过滤。")
    print(f"   全程没有 IoU、没有 NMS，靠的是 DETR 式的**一对一匹配**训练。")

    print(f"\n   框的回归参数化（不是直接回归 9 个数）：")
    print(f"     (x, y, z) 是相对**迭代参考点**的 (0,1) 归一化偏移，配合 inverse_sigmoid 累加")
    print(f"     w, l, h  是 **log 尺寸**，解码时 exp 回来")
    print(f"     yaw      存成 **(sin, cos) 一对**，解码时 atan2 —— 避免角度回归的缠绕问题")
    print(f"     (vx, vy) 不归一化，直接回归")
    print(f"\n   还有一个坑：网络内部 z 是**框中心**，但输出 artifact 时敲掉半个高度")
    print(f"     bboxes[:, 2] -= bboxes[:, 5] * 0.5")
    print(f"   变成**框底**的 z。看代码时如果不注意这一步，会以为高度算错了。")

    banner("⑥ 三个必须知道的坑")
    print(f"   1. **发布的 CUDA deformable-attention kernel 要求 bfloat16**")
    print(f"      `multi_scale_deformable_attn_cuda` 在 CUDA 上遇到非 bf16 会直接抛 TypeError；")
    print(f"      off-GPU 路径有 torch fallback，可用于功能验证，但性能和支持范围不同。")
    print(f"\n   2. **视锥网格故意不注册成 buffer**")
    print(f"      `self.frustum` 是个普通属性，不是 register_buffer。")
    print(f"      原因是 `model.to(bfloat16)` 只转 buffer，普通属性不受影响 —— ")
    print(f"      这样视锥坐标能一直保持 fp32。改成 buffer 会让反投影精度掉下来。")
    print(f"\n   3. **坐标系是 ego，不是 lidar**")
    print(f"      参考点和 BEV 网格都在 ego 系，所以反投影时要显式乘 `inv(lidar2ego)` 绕回来。")
    print(f"      代码里 `lidar2img @ inv(lidar2ego)` 出现的地方就是这个原因。")

    banner("⑦ 和规划分支的关系：并列，不是串联")
    print(f"   {'':<16} {'读 VLM 的什么':<28} {'自己的参数':<16} {'输出'}")
    print(f"   {'-' * 74}")
    print(f"   {'规划专家':<16} {'attention cache (K/V)':<28} {'约 1.0 B':<16} {'50×3 轨迹'}")
    print(f"   {'感知头':<16} {'hidden states + ViT patches':<28} {'约 0.25 B':<16} {'框/占据/地图'}")
    print(f"\n   两者**不共享参数**，也不互相喂数据。")
    print(f"   所以：感知头预测得准，**不意味着**规划用到了那些信息；")
    print(f"   反过来，规划得好也不意味着感知头准确。它们只是共享同一个 VLM 表征。")
    print(f"\n   官方说法是感知头「provides an explicit, inspectable interface to 3D")
    print(f"   scene structure」—— 它的价值之一正是**可检查**：")
    print(f"   规划输出只有一条轨迹，而感知输出是一个可以画出来、和真值对比的三维场景。")


if __name__ == "__main__":
    main()
