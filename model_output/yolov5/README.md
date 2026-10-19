# YOLOv5su → RDK X5

| 项 | 值 |
|---|---|
| 产物 | `yolov5su_640x640_nv12.bin`（11,016,103 字节，sha256 `e32487d4693840501544f6c7a0d27cf41116a316095ae73e0d706327517304b8`） |
| 模型 | YOLOv5su（ultralytics 维护的 anchor-free YOLOv5s），9.14M 参数 / 24.0 GFLOPs |
| 输入 | `images`，**nv12** 运行时 / rgb 训练，NCHW **1×3×640×640**，nv12 缓冲 **614400** 字节 |
| 输出 | `output0` **(1, 84, 8400)** float32 = 4(xywh) + 80(cls)，类别已 sigmoid、框已解码到 640×640 |
| 归一化 | mean 0 / scale 1/255（已烘进模型） |
| 校准 | 128 张 coco128，batch 1，`max`，`max_percentile 0.99996` |
| 实测性能 | subgraph_0 97.6 FPS/10.24ms + subgraph_1 304 FPS/3.29ms → **≈13.5 ms ≈ 74 FPS** |
| 浮点验证 | PyTorch ↔ ONNX：余弦 **1.00000000**，`mean|Δ|=1.54e-06`，top-20 锚点框偏差 **0.000000** |

## 流水线

```bash
# 01 下载官方仓库 + 权重，并导出 ONNX（opset 11）
bash 01_download_and_export.sh

# 02 ONNX 模型验证：PyTorch ↔ ONNX 数值一致性
python3 02_verify_onnx.py

# 03 ONNX → .bin（容器内 checker + makertbin，默认 batch=1 校准）
bash 03_onnx2bin.sh          # 配置见 yolov5su_config.yaml

# 04 .bin 结构与参数验证（主机，hb_model_info）
bash 04_verify_bin_host.sh /open_explorer/_project/model_output/yolov5/yolov5su_640x640_nv12.bin

# 05 板端性能验证（延迟/FPS/输出统计）
python3 05_verify_perf_board.py --runs 30

# 06→07 板端精度验证（先主机生成浮点参考）
python3 06_gen_reference.py --onnx ../yolov5/yolov5su.onnx --mode nv12 --size 640x640 \
    --images /path/a.jpg /path/b.jpg --out ref
python3 07_verify_accuracy_board.py --model ./yolov5su_640x640_nv12.bin --ref ./ref
```

## 后处理要点

`output0` → `转置成 (8400,84)` → `按置信度阈值筛选` → `NMS`。无需 objectness（YOLOv5u 是 anchor-free + 类别分数）。

## 注意

- 该模型输入固定 batch=1，hmct 默认会改写成 batch=8 校准导致 ORT 拒绝，
  所以转换时先跑 `00_toolchain_batch1_patch.py` 把校准批大小固定为 1（`03_onnx2bin.sh` 已包含）。
- ONNX 算子全是 Conv/Mul/Sigmoid/Concat/Resize 等 BPU 原生算子，无风险算子。
