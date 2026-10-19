# MonoOcc R50 主干（ResNet50-C4 + FPN, 384×1280, nv12）→ RDK X5

| 项 | 值 |
|---|---|
| 产物 | `monoocc_r50_trunk_384x1280_nv12.bin`（9,570,609 字节，sha256 `9dd6e8c51cc926f21494a1ccb714d02cfa92766157c3a88cef4eba8603043372`） |
| 权重 | `MonoOcc-S.pth`（557.5 MB，sha256 `4f59f0e2…b6e97`，`state_dict` 513 张量 / 59.03M 参数） |
| 输入 | `images`，**nv12** 运行时 / rgb 训练，NCHW **1×3×384×1280**，nv12 缓冲 **737280** 字节 |
| 输出 | `feat` **1×128×24×80** float32（C4 特征，1/16 分辨率） |
| 归一化 | ImageNet：mean `123.675/116.28/103.53`，scale `1/(255·std)` |
| 校准 | 128 张 coco128，batch 1，`max` |
| 实测性能 | **52.7 FPS / 19.0 ms**，DDR 113 MB |
| 浮点验证 | PyTorch ↔ ONNX：`max|Δ|=3.2e-06`，余弦 **1.00000000** |

## 重要范围说明

这里只导出**主干**（ResNet50-C4 + FPN）。完整 MonoOcc 的 3D 占据头**无法上 BPU**：

- 核心是 `MSDeformableAttention3D`（仓库 `ops/` 里是编译好的 CUDA 扩展 `.so`），BPU 无此算子；
- BEVFormer 式**数据相关采样**只能落 CPU；
- 输出 BEV 128×128×16 × 20 类 = 5.24M logits（约 21 MB），不适合做 BPU 输出；
- 头部还需相机内参参与运算（动态图）。

所以正确形态是：**BPU 跑主干（19 ms）→ CPU 跑 3D 头**。

另一个关键点：不需要 mmcv/mmdet/mmdetd3 全栈。`02_pth2onnx.py` 用
**torchvision ResNet50 + 自写 FPN** 从 checkpoint 精确加载权重（318 个 backbone 张量严格匹配，仅缺 `fc`），
FPN 就是 `Conv1x1(1024→128) → Conv3x3(128→128)` 且无激活（与 mmcv FPN `num_outs=1, act_cfg=None` 等价）。

## 流水线

```bash
# 01 准备权重（从 Windows 下载目录拷贝，或按提示从官方 Model Zoo 下载）+ 仓库
bash 01_download_weights.sh

# 02 pth → onnx（torchvision R50 + FPN，opset 11）
python3 02_pth2onnx.py --ckpt /path/MonoOcc-S.pth -o monoocc_r50_trunk.onnx --height 384 --width 1280

# 03 ONNX 模型验证（PyTorch ↔ ONNX，含 RT-DETR 部分，可只看 B 段）
python3 03_verify_onnx.py

# 04 onnx → bin
bash 04_onnx2bin.sh          # 配置见 monoocc_r50_trunk_config.yaml

# 05 结构验证（主机）
bash 05_verify_bin_host.sh /open_explorer/_project/model_output/monoocc_r50_trunk/monoocc_r50_trunk_384x1280_nv12.bin

# 06 板端性能验证
python3 06_verify_perf_board.py --runs 20

# 07→08 板端精度验证
python3 07_gen_reference.py --onnx ./monoocc_r50_trunk.onnx --mode nv12 --size 384x1280 \
    --images /path/a.jpg --out ref \
    --mean 123.675,116.28,103.53 --scale 0.01712475383,0.01750700281,0.0174291939
python3 08_verify_accuracy_board.py --model ./monoocc_r50_trunk_384x1280_nv12.bin --ref ./ref
```

## 注意

- 数据集的真实输入是 **370×1220**（`semantic_kitti_dataset_stage2.py:69-70`），我按 **384×1280**（32 的整数倍、对 BPU 友好）建库；
  若要接 3D 头，需按 370×1220 重导（C4 会变成 23×76）。
- 输入固定 batch=1，`04_onnx2bin.sh` 先跑 `00_toolchain_batch1_patch.py`。
