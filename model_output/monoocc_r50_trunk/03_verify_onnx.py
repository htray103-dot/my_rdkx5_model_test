#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""浮点级一致性验证（在宿主机 WSL 上跑，不需要板子）

A) RT-DETR：我构造的 opset11 单输入 ONNX  vs  官方导出（opset16，torch 2.3.1）
   —— 验证我用 [1,1,2,2] 替换空 scales 的等价改写没有改变语义。

B) MonoOcc R50 主干：torch 前向  vs  导出的 ONNX（onnxruntime）
   —— 验证 ONNX 导出与权重加载正确。
"""
import numpy as np
import torch

RT_OP11 = "/home/weydad/open_explorer/_project/DETR/rtdetr/rtdetrv2_r18vd_120e_coco_opset11.onnx"
RT_OP16 = "/home/weydad/open_explorer/_project/DETR/RT-DETR/rtdetrv2_pytorch/rtdetrv2_r18vd_120e_coco.onnx"
MONO_ONNX = "/home/weydad/open_explorer/_project/Mono_Occupancy/R50/monoocc_r50_trunk.onnx"
MONO_CKPT = "/home/weydad/open_explorer/_project/Mono_Occupancy/R50/MonoOcc-S.pth"


def stat(name, a, b):
    a = np.asarray(a).astype(np.float64).ravel()
    b = np.asarray(b).astype(np.float64).ravel()
    if a.size != b.size:
        print("  %-8s 形状不一致: %s vs %s" % (name, a.shape, b.shape))
        return
    diff = np.abs(a - b)
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    rel = float(diff.mean() / (np.abs(a).mean() + 1e-12))
    print("  %-8s max|Δ|=%.6g  mean|Δ|=%.6g  相对误差=%.3e  余弦相似度=%.8f" % (
        name, diff.max(), diff.mean(), rel, cos))


print("=" * 78)
print("A) RT-DETR: opset11（我的改写） vs opset16（官方导出）")
print("=" * 78)
import onnxruntime as ort

s11 = ort.InferenceSession(RT_OP11, providers=["CPUExecutionProvider"])
s16 = ort.InferenceSession(RT_OP16, providers=["CPUExecutionProvider"])
print("opset11 输入:", [(i.name, i.shape) for i in s11.get_inputs()])
print("opset16 输入:", [(i.name, i.shape) for i in s16.get_inputs()])

rng = np.random.default_rng(2026)
for trial in range(3):
    img = rng.random((1, 3, 640, 640), dtype=np.float32)
    out11 = s11.run(None, {s11.get_inputs()[0].name: img})
    out16 = s16.run(None, {s16.get_inputs()[0].name: img,
                           s16.get_inputs()[1].name: np.array([[640, 640]], dtype=np.int64)})
    print("-- 随机样本 #%d" % (trial + 1))
    for i, nm in enumerate(["labels", "boxes", "scores"]):
        stat(nm, out16[i], out11[i])
    # labels 完全一致的比例
    lab_eq = float((np.asarray(out16[0]) == np.asarray(out11[0])).mean())
    print("  labels 完全相等比例 = %.4f" % lab_eq)

print()
print("=" * 78)
print("B) MonoOcc R50 主干：torch 前向 vs ONNX（onnxruntime）")
print("=" * 78)
import torch.nn as nn
from torchvision.models import resnet50


class FPN1(nn.Module):
    def __init__(self):
        super().__init__()
        self.lateral_conv = nn.Conv2d(1024, 128, 1)
        self.fpn_conv = nn.Conv2d(128, 128, 3, padding=1)

    def forward(self, c4):
        return self.fpn_conv(self.lateral_conv(c4))


class Trunk(nn.Module):
    def __init__(self, r, f):
        super().__init__()
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool,
                                  r.layer1, r.layer2, r.layer3)
        self.fpn = f

    def forward(self, x):
        return self.fpn(self.stem(x))


ck = torch.load(MONO_CKPT, map_location="cpu", weights_only=False)["state_dict"]
r50 = resnet50(weights=None)
r50.load_state_dict({k[len("img_backbone."):]: v for k, v in ck.items()
                     if k.startswith("img_backbone.")}, strict=False)
fpn = FPN1()
with torch.no_grad():
    fpn.lateral_conv.weight.copy_(ck["img_neck.lateral_convs.0.conv.weight"])
    fpn.lateral_conv.bias.copy_(ck["img_neck.lateral_convs.0.conv.bias"])
    fpn.fpn_conv.weight.copy_(ck["img_neck.fpn_convs.0.conv.weight"])
    fpn.fpn_conv.bias.copy_(ck["img_neck.fpn_convs.0.conv.bias"])
model = Trunk(r50, fpn).eval()

sess = ort.InferenceSession(MONO_ONNX, providers=["CPUExecutionProvider"])
iname = sess.get_inputs()[0].name
print("ONNX 输入:", iname, sess.get_inputs()[0].shape)

for trial in range(2):
    x = rng.standard_normal((1, 3, 384, 1280), dtype=np.float32)
    with torch.no_grad():
        y_t = model(torch.from_numpy(x)).numpy()
    y_o = sess.run(None, {iname: x})[0]
    print("-- 随机样本 #%d  shape=%s" % (trial + 1, y_o.shape))
    stat("feat", y_t, y_o)

print()
print("完成。")
