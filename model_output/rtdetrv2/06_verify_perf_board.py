#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RDK X5 板端验证脚本 —— RT-DETRv2-S (r18vd, 640x640, rgb)

用法（在板子上）:
    python3 verify_rtdetrv2.py
    python3 verify_rtdetrv2.py --runs 20 --image demo.npy   # 喂 NCHW float32 (1,3,640,640)

模型 I/O:
    input : images  rgb  NCHW  1x3x640x640   (归一化 mean=0, scale=1/255 已在模型内)
    outputs: labels [1,300] / boxes [1,300,4] / scores [1,300]
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

DEFAULT_MODEL = "/root/rtdetrv2_r18vd_640x640_rgb.bin"


def describe(model, idx=0):
    print("[submodel %d]" % idx)
    for t in model.inputs:
        p = t.properties
        print("  input : %-12s shape=%s dtype=%s" % (t.name, p.shape, p.dtype))
    for t in model.outputs:
        p = t.properties
        print("  output: %-12s shape=%s dtype=%s" % (t.name, p.shape, p.dtype))


def make_dummy(model):
    t = model.inputs[0]
    p = t.properties
    shape = tuple(p.shape)
    dt = p.dtype if p.dtype is not None else np.float32
    # 若是 HWC uint8（rgb 直通）就按 HWC 造，否则按属性原样
    if len(shape) == 3 and shape[2] == 3 and dt == np.uint8:
        return [np.zeros(shape, dtype=np.uint8)]
    return [np.zeros(shape, dtype=dt)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--image", default=None, help="可选: .npy 文件，形状 1x3x640x640 float32")
    args = ap.parse_args()

    if not os.path.exists(args.model):
        print("✘ 找不到模型: %s" % args.model)
        return 1
    print("模型:", args.model, "大小 %.1f MB" % (os.path.getsize(args.model) / 1024 / 1024))
    models = dnn.load(args.model)
    print("子模型数:", len(models))
    describe(models[0])

    if args.image:
        inputs = [np.load(args.image).astype(np.float32)]
    else:
        inputs = make_dummy(models[0])
        print("使用 dummy 输入:", [i.shape for i in inputs])
    print("输入:", [(i.shape, str(i.dtype)) for i in inputs])

    outs = models[0].forward(inputs)
    print("\n输出:")
    names = ["labels", "boxes", "scores"]
    for i, o in enumerate(outs):
        a = np.asarray(o.buffer)
        print("  %-8s shape=%-14s dtype=%-8s min=%.4f max=%.4f" % (
            names[i] if i < len(names) else str(i), a.shape, a.dtype,
            float(np.min(a)), float(np.max(a))))
        if i == 2:  # scores
            k = int(np.argmax(a))
            print("    top-1 score=%.4f  index=%d" % (float(a.ravel()[k]), k))

    lat = []
    for _ in range(max(1, args.runs)):
        t0 = time.time()
        models[0].forward(inputs)
        lat.append((time.time() - t0) * 1000.0)
    lat = np.array(lat)
    print("\n延迟(ms): min=%.2f  中位=%.2f  平均=%.2f  max=%.2f" % (
        lat.min(), float(np.median(lat)), lat.mean(), lat.max()))
    print("折算 FPS(中位): %.1f" % (1000.0 / float(np.median(lat))))
    print("\nNOTE: boxes 是 640x640 坐标系（模型内置该尺寸常量），上屏前按实际图像比例缩放。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
