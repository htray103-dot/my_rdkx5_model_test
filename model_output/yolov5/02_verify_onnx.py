#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLOv5su 导出保真度验证：ultralytics PyTorch 前向 vs 导出的 ONNX

两者喂同一张真实图片（RGB, resize 640, /255），比较 (1,84,8400) 原始输出。
"""
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

PT = "/home/weydad/open_explorer/_project/yolov5/yolov5su.pt"
ONNX = "/home/weydad/open_explorer/_project/yolov5/yolov5su.onnx"
IMG = "/home/weydad/open_explorer/_project/cal_data_dir/coco/coco128/images/train2017/000000000009.jpg"

y = YOLO(PT)
net = y.model.float().eval()
if hasattr(net, "fuse"):
    net = net.fuse()

im = Image.open(IMG).convert("RGB").resize((640, 640), Image.BILINEAR)
x = torch.from_numpy(np.transpose(np.asarray(im, dtype=np.float32) / 255.0, (2, 0, 1)))[None]

with torch.no_grad():
    o = net(x)
t_out = o[0] if isinstance(o, (tuple, list)) else o
t_out = t_out.numpy()

import onnxruntime as ort

sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
o_out = sess.run(None, {sess.get_inputs()[0].name: x.numpy()})[0]

print("图片:", IMG.split("/")[-1])
print("PyTorch 输出:", t_out.shape, t_out.dtype)
print("ONNX    输出:", o_out.shape, o_out.dtype)
if t_out.shape != o_out.shape:
    print("✘ 形状不一致，无法逐元素比较")
    raise SystemExit(1)

a = t_out.astype(np.float64)
b = o_out.astype(np.float64)
d = np.abs(a - b)
cos = float(np.dot(a.ravel(), b.ravel()) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
print("\nmax|Δ|=%.6g  mean|Δ|=%.6g  余弦相似度=%.8f" % (d.max(), d.mean(), cos))

# 对“有意义”的部分比较：置信度最高的 top-20 锚点
C = a.shape[1]                      # 84
N = a.shape[2]                      # 8400
A2 = a.reshape(C, N)
B2 = b.reshape(C, N)
conf_t = A2[4:, :].max(axis=0)
conf_o = B2[4:, :].max(axis=0)
top = np.argsort(-conf_t)[:20]
box_err = np.abs(A2[:4, top] - B2[:4, top]) / 640.0
print("top-20 锚点框相对偏差(N/640): mean=%.6f  max=%.6f" % (box_err.mean(), box_err.max()))
print("top-20 锚点置信度最大差: %.6f" % float(np.abs(conf_t[top] - conf_o[top]).max()))
print("\n结论: 若余弦≈1 且 top 框偏差≈0，则 ONNX 导出保真，可用于后续量化。")
