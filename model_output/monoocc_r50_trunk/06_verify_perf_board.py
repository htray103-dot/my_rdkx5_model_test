#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RDK X5 板端验证脚本 —— MonoOcc R50 主干 (384x1280, nv12)

用法（在板子上）:
    python3 verify_monoocc_trunk.py
    python3 verify_monoocc_trunk.py --runs 20 --yuv demo.yuv

模型 I/O:
    input : images  nv12  1x3x384x1280   (nv12 缓冲 = 384*1280*3/2 = 737280 字节)
    output: feat    1x128x24x80 float32  (C4 特征，1/16 分辨率)

注意: 这只是 MonoOcc 的**特征提取主干**；3D 占据头（BEVFormer 式 deformable attention）
      无法上 BPU，需在 CPU/GPU 侧运行。
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

DEFAULT_MODEL = "/root/monoocc_r50_trunk_384x1280_nv12.bin"
NV12_BYTES = 384 * 1280 * 3 // 2   # 737280


def describe(model, idx=0):
    print("[submodel %d]" % idx)
    for t in model.inputs:
        p = t.properties
        print("  input : %-10s shape=%s dtype=%s" % (t.name, p.shape, p.dtype))
    for t in model.outputs:
        p = t.properties
        print("  output: %-10s shape=%s dtype=%s" % (t.name, p.shape, p.dtype))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--yuv", default=None, help="可选：真实 nv12 文件（长度 737280）")
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
        inputs = [np.zeros(NV12_BYTES, dtype=np.uint8)]
        print("使用全零 nv12 输入:", inputs[0].shape)

    outs = models[0].forward(inputs)
    print("\n输出:")
    for i, o in enumerate(outs):
        a = np.asarray(o.buffer)
        print("  out[%d] shape=%-16s dtype=%-8s min=%.4f max=%.4f mean=%.4f std=%.4f" % (
            i, a.shape, a.dtype, float(a.min()), float(a.max()),
            float(a.mean()), float(a.std())))

    lat = []
    for _ in range(max(1, args.runs)):
        t0 = time.time()
        models[0].forward(inputs)
        lat.append((time.time() - t0) * 1000.0)
    lat = np.array(lat)
    print("\n延迟(ms): min=%.2f  中位=%.2f  平均=%.2f  max=%.2f" % (
        lat.min(), float(np.median(lat)), lat.mean(), lat.max()))
    print("折算 FPS(中位): %.1f" % (1000.0 / float(np.median(lat))))
    print("\nNOTE: 输出是 C4 特征图 (128x24x80)，后续 3D 占据头需另行实现。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
