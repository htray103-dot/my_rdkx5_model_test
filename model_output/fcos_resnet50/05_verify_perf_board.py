#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RDK X5 板端验证脚本 —— FCOS-ResNet50 (1024x1024, nv12)

用法（在板子上）:
    python3 verify_fcos_resnet50.py                 # 自动探测输入，用全零数据跑通
    python3 verify_fcos_resnet50.py --runs 20       # 跑 20 次取延迟中位数
    python3 verify_fcos_resnet50.py --yuv demo.yuv  # 喂真实 nv12（长度 1024*1024*3/2 = 1572864）

模型信息:
    input : input.1  nv12  1x3x1024x1024   (mean=128, scale=1/128 已烘进模型)
    outputs: 15 个张量 = 5 个尺度 × (cls 80 类 / reg 4 / centerness 1)
             尺寸 (1,80,128,128) (1,80,64,64) (1,80,32,32) (1,80,16,16) (1,80,8,8) 及其 reg/ctr 对应层
    后处理需自行实现: sigmoid → 按 stride 解码 → NMS
"""
import argparse
import os
import sys
import time

import numpy as np

try:
    from hobot_dnn import pyeasy_dnn as dnn
except Exception as e:
    print("✘ 需要在 RDK X5 板端运行（hobot_dnn 不可用）: %s" % e)
    sys.exit(1)

DEFAULT_MODEL = "/root/fcos_resnet50_1024x1024_nv12.bin"
NV12_BYTES = 1024 * 1024 * 3 // 2      # 1572864


def describe(model, idx=0):
    print("[submodel %d]" % idx)
    for t in model.inputs:
        p = t.properties
        print("  input : %-10s shape=%s dtype=%s" % (t.name, p.shape, p.dtype))
    for t in model.outputs:
        p = t.properties
        print("  output: %-10s shape=%s dtype=%s" % (t.name, p.shape, p.dtype))


def make_dummy(model):
    """按输入属性造一组合法输入；nv12 模型给 H*W*3/2 字节的一维缓冲。"""
    p = model.inputs[0].properties
    shape = tuple(int(x) for x in p.shape)
    dt = p.dtype if p.dtype is not None else np.float32
    if len(shape) == 3 and shape[2] == 3 and dt == np.uint8:
        h, w = shape[0], shape[1]
        return [np.zeros(h * w * 3 // 2, dtype=np.uint8)]
    return [np.zeros(shape, dtype=dt)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--yuv", default=None, help="可选：真实 nv12 文件")
    args = ap.parse_args()

    if not os.path.exists(args.model):
        print("✘ 找不到模型: %s" % args.model)
        return 1
    print("模型:", args.model, "大小 %.1f MB" % (os.path.getsize(args.model) / 1024 / 1024))

    models = dnn.load(args.model)
    print("子模型数:", len(models))
    describe(models[0])

    if args.yuv:
        raw = np.fromfile(args.yuv, dtype=np.uint8)
        print("nv12 文件: %d 字节（期望 %d）" % (raw.size, NV12_BYTES))
        inputs = [raw]
    else:
        inputs = make_dummy(models[0])
        print("dummy 输入:", [(i.shape, str(i.dtype)) for i in inputs])

    outs = models[0].forward(inputs)
    print("\n输出张量数:", len(outs))
    for i, o in enumerate(outs):
        a = np.asarray(o.buffer)
        print("  out[%2d] shape=%-16s dtype=%-8s min=%.3f max=%.3f" % (
            i, a.shape, a.dtype, float(a.min()), float(a.max())))

    lat = []
    for _ in range(max(1, args.runs)):
        t0 = time.time()
        models[0].forward(inputs)
        lat.append((time.time() - t0) * 1000.0)
    lat = np.array(lat)
    print("\n延迟(ms): min=%.2f 中位=%.2f 平均=%.2f max=%.2f" % (
        lat.min(), float(np.median(lat)), lat.mean(), lat.max()))
    print("折算 FPS(中位): %.1f" % (1000.0 / float(np.median(lat))))
    print("\nNOTE: 本脚本验证“能跑通 + 测速”；检测框需配合 sigmoid/decode/NMS 后处理。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
