# FCOS-ResNet50 (1024×1024, nv12) → RDK X5

| 项 | 值 |
|---|---|
| 产物 | `fcos_resnet50_1024x1024_nv12.bin`（45,095,803 字节，sha256 `55807779792455d47dd12ce21ea914d66713a48154d02c96977de9a9e99c993c`） |
| 来源 | SDK 自带模型：`horizon_model_convert_sample/01_common/model_zoo/mapper/detection/fcos_resnet50/fcos_resnet50.onnx`（opset 11，277 节点，215 MB） |
| 输入 | `input.1`，**nv12** 运行时 / **yuv444** 训练，NCHW **1×3×1024×1024**，nv12 缓冲 **1572864** 字节 |
| 输出 | 15 个张量 = 5 尺度 × (cls 80 / reg 4 / centerness 1)，尺度 128/64/32/16/8 |
| 归一化 | mean 128 / scale 1/128（已烘进模型） |
| 校准 | 50 张 SDK 自带 `.yuv`（float32，`calibration_data_yuv_f32`），`max`，`max_percentile 0.999997` |
| 编译报告 | 单子图 ≈8.5 FPS / 117.6 ms / DDR 402 MB |

## 关键修复：Resize 的空 scales

原 ONNX 的两个 `Resize` 节点带一个**空常量 scales**（`dims: 0`），ONNX 里表示“不提供”，
但 onnxruntime 会把它当成长度 0 的 scales 做校验并报错：

```
'Linear' mode and 'Cubic' mode only support 2-D inputs or 4-D inputs with
the corresponding outermost 2 scale values being 1 in the Resize operator
```

`01_extract_onnx.py` 的处理：把空 scales 换成真实 **`[1,1,2,2]`**（该 neck 固定 ×2 上采样，
由形状推断确认输入/输出为 32→64、64→128），并省略 `sizes` 输入。语义不变，ORT 前向随即通过。

> 这也是之前只能用 `--fast-perf`（`calibration_type: skip`）出 .bin 的根本原因。

## 流水线

```bash
# 01 取 SDK 的 ONNX + 修 Resize + ORT 前向自检（产物 fcos_resnet50_resize_fix.onnx）
python3 01_extract_onnx.py

# 02 ONNX 模型验证（checker + 15 输出形状/尺度/finite）
python3 02_verify_onnx.py fcos_resnet50_resize_fix.onnx

# 03 onnx → bin（batch=1 校准 + checker + makertbin）
bash 03_onnx2bin.sh          # 配置见 fcos_resnet50_config.yaml

# 04 结构验证（主机）
bash 04_verify_bin_host.sh /open_explorer/_project/model_output/fcos_resnet50/fcos_resnet50_1024x1024_nv12.bin

# 05 板端性能验证
python3 05_verify_perf_board.py --runs 10

# 06→07 板端精度验证
python3 06_gen_reference.py --onnx fcos_resnet50_resize_fix.onnx --mode nv12 --size 1024x1024 \
    --images /path/a.jpg --out ref --mean 128,128,128 --scale 0.0078125,0.0078125,0.0078125
python3 07_verify_accuracy_board.py --model ./fcos_resnet50_1024x1024_nv12.bin --ref ./ref
```

## 注意

- 该模型输入固定 batch=1，`03_onnx2bin.sh` 会先跑 `00_toolchain_batch1_patch.py`。
- 校准数据用 SDK 自带的 float32 `.yuv`（与 `input_type_train=yuv444`、NCHW 一致），
  `preprocess_on` 必须为 `False`（走二进制读取），否则会把 `.yuv` 当图片解码。
- 后处理：5 尺度 × (cls/reg/ctr) → sigmoid → 按 stride 解码 → NMS。
