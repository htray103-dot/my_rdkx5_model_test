# model_output —— RDK X5 可部署模型总览

四个已转换并验证的模型，每个模型一个子目录，**`.bin` + 完整转化流水线脚本**都在其目录内。

| 子目录 | 模型 | .bin | 大小 | 输入 | 实测性能 |
|---|---|---|---|---|---|
| `yolov5/` | YOLOv5su（anchor-free，9.14M 参数 / 24 GFLOPs） | `yolov5su_640x640_nv12.bin` | 11.0 MB | nv12 **640×640** | **≈13.5 ms ≈ 74 FPS** |
| `rtdetrv2/` | RT-DETRv2-S (r18vd, 20.25M) | `rtdetrv2_r18vd_640x640_rgb.bin` | 39.8 MB | rgb **640×640** | ≈482 ms ≈ 2 FPS |
| `fcos_resnet50/` | FCOS-ResNet50 (1024×1024) | `fcos_resnet50_1024x1024_nv12.bin` | 45.1 MB | nv12 **1024×1024** | 编译报告中 ~8.5 FPS |
| `monoocc_r50_trunk/` | MonoOcc 主干 R50-C4+FPN | `monoocc_r50_trunk_384x1280_nv12.bin` | 9.6 MB | nv12 **384×1280** | ≈19 ms ≈ 53 FPS |

全部产物：`BPU march = bayes-e`、`hb_mapper 1.24.3 / hbdk 3.49.15`、INT8 量化。

## 统一流水线（各目录内脚本编号对应）

| 阶段 | 脚本 | 运行环境 |
|---|---|---|
| 00 | `00_toolchain_batch1_patch.py` — 把 hmct 校准批大小固定为 1（固定 batch=1 的模型必需） | 工具链容器内 |
| 01 | `01_download*` — 下载/准备权重与代码 | WSL 宿主机 |
| 02 | `02_pth2onnx*` — PyTorch → ONNX（opset 11） | WSL 宿主机 |
| 03 | `0x_verify_onnx*` — ONNX 模型验证（PyTorch↔ONNX 数值一致性 / 功能自检） | WSL 宿主机 |
| 04 | `0x_onnx2bin*` — hb_mapper checker + makertbin → `.bin` | 工具链容器 |
| 05 | `0x_verify_bin_host.sh` — `.bin` 结构与参数验证（hb_model_info） | 工具链容器 |
| 06 | `0x_verify_perf_board.py` — **性能验证**：延迟/FPS/输出张量（需 hobot_dnn） | **RDK X5 板端** |
| 07 | `0x_gen_reference.py` — **精度验证·第1步**：用浮点 ONNX 生成参考输出到 `ref/` | WSL 宿主机 |
| 08 | `0x_verify_accuracy_board.py` — **精度验证·第2步**：`.bin` 输出 vs 浮点参考，算余弦/误差 | **RDK X5 板端** |

> 编号在不同模型里略有偏移（YOLOv5 把下载与导出合成 `01`；FCOS 没有 pth→onnx，`01` 是从 SDK 取 ONNX 并修补），各目录 README 里有具体说明。

## 快速使用

**主机端（不用上板）验证结构**
```bash
bash yolov5/04_verify_bin_host.sh "$(pwd)/yolov5/yolov5su_640x640_nv12.bin"
```

**板端性能验证**
```bash
scp yolov5/yolov5su_640x640_nv12.bin yolov5/05_verify_perf_board.py root@<板IP>:/root/
python3 /root/05_verify_perf_board.py --runs 30
```

**板端精度验证（两步）**
```bash
# 1) 主机：生成浮点参考（输入类型/尺寸跟 .bin 对齐）
cd yolov5 && python3 06_gen_reference.py --onnx ../../yolov5/yolov5su.onnx \
    --mode nv12 --size 640x640 --images <若干 jpg> --out ref
# 2) 板端：比对
scp -r ref 07_verify_accuracy_board.py root@<板IP>:/root/
python3 /root/07_verify_accuracy_board.py --model /root/yolov5su_640x640_nv12.bin --ref /root/ref
```

## 已验证结论（浮点层面）

| 模型 | 验证方式 | 结果 |
|---|---|---|
| YOLOv5su | PyTorch ↔ ONNX（真实图片） | 余弦 **1.00000000**，top-20 框偏差 **0.000000** |
| RT-DETRv2 | PyTorch ↔ ONNX（同权重） | scores 余弦 **1.00000000**，top-10 框偏差 **0.000000** |
| MonoOcc 主干 | PyTorch ↔ ONNX | `max|Δ|=3.2e-06`，余弦 **1.00000000** |
| FCOS | ONNX 功能自检（15 输出、5 尺度、finite） | PASS |

INT8 精度（`.bin` vs 浮点）需按上面“精度验证”两步在板上实测。
