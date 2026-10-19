#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RDK X5 板端验证脚本 —— YOLOv5su (640x640, nv12)

用法（在板子上）:
    python3 verify_yolov5su.py
    python3 verify_yolov5su.py --runs 30
    python3 verify_yolov5su.py --yuv demo.yuv      # 真实 nv12（长度 640*640*3/2 = 614400）

模型 I/O:
    input : images  nv12  1x3x640x640   (rgb 训练侧, /255 已烘进模型, nv12 buffer = 614400 字节)
    output: output0  (1, 84, 8400) float32
            84 = 4(xywh) + 80(classes)；classes 已 sigmoid，框已解码到 640x640 坐标系
            后处理：转置成 (8400,84) → 按置信度阈值筛选 → NMS
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

DEFAULT_MODEL = "/root/yolov5su_640x640_nv12.bin"
NV12_BYTES = 640 * 640 * 3 // 2      # 614400


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
    ap.add_argument("--yuv", default=None)
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
        print("dummy nv12 输入:", inputs[0].shape)

    outs = models[0].forward(inputs)
    print("\n输出:")
    for i, o in enumerate(outs):
        a = np.asarray(o.buffer)
        print("  out[%d] shape=%-16s dtype=%-8s min=%.4f max=%.4f mean=%.4f" % (
            i, a.shape, a.dtype, float(a.min()), float(a.max()), float(a.mean())))
        # 简单统计：按 (84,8400) 解释，看类别分数最大值
        if a.size == 84 * 8400:
            pred = a.reshape(84, 8400)
            cls = pred[4:, :]
            print("    类别分数 max=%.4f (>0.25 的锚点数=%d)" % (
                float(cls.max()), int((cls.max(axis=0) > 0.25).sum())))

    lat = []
    for _ in range(max(1, args.runs)):
        t0 = time.time()
        models[0].forward(inputs)
        lat.append((time.time() - t0) * 1000.0)
    lat = np.array(lat)
    print("\n延迟(ms): min=%.2f 中位=%.2f 平均=%.2f max=%.2f" % (
        lat.min(), float(np.median(lat)), lat.mean(), lat.max()))
    print("折算 FPS(中位): %.1f" % (1000.0 / float(np.median(lat))))
    print("\nNOTE: 本脚本验证“能跑通 + 测速”；检测框需配合阈值筛选 + NMS 后处理。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
