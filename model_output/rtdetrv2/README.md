# RT-DETRv2-S (r18vd, 640×640, rgb) → RDK X5

| 项 | 值 |
|---|---|
| 产物 | `rtdetrv2_r18vd_640x640_rgb.bin`（39,805,887 字节，sha256 `a806c42b14130f4b718f14e1cb09c29b881c74fbd87276a8218b22bedc7695f8`） |
| 权重 | `rtdetrv2_r18vd_120e_coco_rerun_48.1.pth`（77.4 MB，sha256 `2ace5218…c77e`，`ckpt['ema']['module']`，538 张量 / 20.25M） |
| 输入 | `images`，**rgb** 运行时 / rgb 训练，NCHW **1×3×640×640** |
| 输出 | `labels[1,300]` int64 / `boxes[1,300,4]` float32（640 坐标系）/ `scores[1,300]` float32 |
| 归一化 | mean 0 / scale 1/255（已烘进模型） |
| 校准 | 24 张 coco128，batch 1，`max`（fast 配置；想提精度可换 128 张 + `default`/`kl`） |
| 实测性能 | 10 个 BPU 子图，合计 **≈482 ms ≈ 2 FPS**（解码器部分落 CPU，见下） |
| 浮点验证 | 同权重 PyTorch ↔ ONNX：scores 余弦 **1.00000000**、`max|Δ|=6.7e-06`、labels 99.33% 全等、top-10 框偏差 **0.000000** |

## 关键背景：为什么要自己改 ONNX

RT-DETR 系列解码器用到 `grid_sample`，对应 ONNX `GridSample` 算子**要求 opset ≥ 16**，
而地平线 hb_mapper 上限是 **opset 11**（实测报错：`The opset version of the model is 16, the maximum supported version is 11`；
Ultralytics 版 RT-DETR 更直接断言 `RTDETR export requires opset>=16`）。

因此 `02_pth2onnx.sh` 用**等价实现替换**两处算子后导出 opset 11：

- `F.scaled_dot_product_attention` → 显式 `softmax(QKᵀ/√d)V`
- `F.grid_sample`（双线性 + zeros padding + align_corners=False）→ 手写 `gather` 实现
  自检误差 **1.192e-07**（float32 精度极限）

## 流水线

```bash
# 01 下载权重
bash 01_download_weights.sh

# 02 pth → onnx（opset 11，单输入，尺寸常量固化）
bash 02_pth2onnx.sh          # 调用 export_onnx_rdk.py

# 03 ONNX 模型验证（PyTorch ↔ ONNX）
python3 03_verify_onnx.py

# 04 onnx → bin（含 ONNX 整形 make_rdk_onnx.py + checker + makertbin）
bash 04_onnx2bin.sh          # 配置见 rtdetrv2_r18vd_config_fast.yaml

# 05 结构验证（主机）
bash 05_verify_bin_host.sh /open_explorer/_project/model_output/rtdetrv2/rtdetrv2_r18vd_640x640_rgb.bin

# 06 板端性能验证
python3 06_verify_perf_board.py --runs 20

# 07→08 板端精度验证
python3 07_gen_reference.py --onnx ../../DETR/rtdetr/rtdetrv2_r18vd_120e_coco_opset11.onnx \
    --mode rgb --size 640x640 --images /path/a.jpg --out ref
python3 08_verify_accuracy_board.py --model ./rtdetrv2_r18vd_640x640_rgb.bin --ref ./ref
```

## 性能瓶颈说明

编译日志里解码器出现大量 `Can not find the scale for node … GatherElements_Cast`，
说明这些 int64 的 Cast/Gather（我用来等价替换 grid_sample 的算子）**没有量化参数、落到 CPU**，
这既是 2 FPS 的原因，也意味着它们以浮点执行、精度安全。

想提速的可行方向：只把 backbone/encoder 放 BPU、解码器整体移到 CPU（并改成 batch 化实现），
或直接换 YOLO 系列（同类任务下 ≈74 FPS）。
