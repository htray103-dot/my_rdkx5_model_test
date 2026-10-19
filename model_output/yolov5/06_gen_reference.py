#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""【主机端·精度验证第 1 步】用浮点 ONNX 生成参考输出，供板端比对

用法（在 WSL 宿主机跑，需要 onnxruntime + PIL）:
    python3 06_gen_reference.py --onnx yolov5su.onnx --mode nv12 --size 640x640 \
        --images img1.jpg img2.jpg --out ref

产物（写入 --out 目录，默认 ./ref）:
    in_000.npy   : 板端要喂给 .bin 的原始数据（nv12 为一维 uint8；rgb 为 HWC uint8）
    out_000.npz  : 浮点 ONNX 的各输出（键 out0, out1, ...）
    meta.json    : 输入类型/尺寸/图片名
"""
import argparse
import json
import os

import numpy as np
from PIL import Image


def rgb_to_nv12(rgb):  # rgb: HxWx3 uint8，BT.601 有限范围
    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)
    y = 0.257 * r + 0.504 * g + 0.098 * b + 16.0
    u = -0.148 * r - 0.291 * g + 0.439 * b + 128.0
    v = 0.439 * r - 0.368 * g - 0.071 * b + 128.0
    y = np.clip(y, 0, 255).astype(np.uint8)
    u = np.clip(u, 0, 255).astype(np.uint8)
    v = np.clip(v, 0, 255).astype(np.uint8)
    h, w = y.shape
    uv = np.empty((h // 2, w // 2, 2), dtype=np.uint8)
    uv[:, :, 0] = u.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))
    uv[:, :, 1] = v.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))
    return np.concatenate([y.ravel(), uv.ravel()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--mode", choices=["nv12", "rgb"], required=True,
                    help="与 .bin 的 input_type_rt 一致")
    ap.add_argument("--size", required=True, help="HxW，例如 640x640")
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--out", default="ref")
    ap.add_argument("--mean", default="0,0,0")
    ap.add_argument("--scale", default="0.003921568627451,0.003921568627451,0.003921568627451")
    args = ap.parse_args()

    H, W = [int(x) for x in args.size.lower().split("x")]
    mean = np.array([float(x) for x in args.mean.split(",")], dtype=np.float32).reshape(3, 1, 1)
    scale = np.array([float(x) for x in args.scale.split(",")], dtype=np.float32).reshape(3, 1, 1)
    os.makedirs(args.out, exist_ok=True)

    import onnxruntime as ort
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name

    meta = {"mode": args.mode, "size": [H, W], "onnx": args.onnx, "items": []}
    for i, p in enumerate(args.images):
        rgb = np.asarray(Image.open(p).convert("RGB").resize((W, H), Image.BILINEAR))
        # 板端喂的原始数据（rt 侧）
        rt_input = rgb_to_nv12(rgb) if args.mode == "nv12" else rgb
        # ONNX 侧的输入（train 侧，float NCHW）——与 .bin 内部预处理等价
        x = np.transpose(rgb.astype(np.float32), (2, 0, 1))[None]      # 0-255
        x = (x - mean[None]) * scale[None]                             # 归一化
        outs = sess.run(None, {iname: x})
        np.save(os.path.join(args.out, "in_%03d.npy" % i), rt_input)
        np.savez(os.path.join(args.out, "out_%03d.npz" % i),
                 **{"out%d" % k: np.asarray(v) for k, v in enumerate(outs)})
        meta["items"].append({"index": i, "image": os.path.basename(p),
                              "n_outputs": len(outs),
                              "shapes": [list(np.asarray(v).shape) for v in outs]})
        print("  [%d] %s -> rt_input %s (%s), %d 个输出" % (
            i, os.path.basename(p), rt_input.shape, rt_input.dtype, len(outs)))

    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print("参考数据已写入:", args.out)


if __name__ == "__main__":
    main()
