#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""【板端·精度验证】.bin(INT8) 输出 与 浮点 ONNX 参考输出 逐元素比对

前置：在 WSL 宿主机用 06_gen_reference.py 生成 ref/ 目录（含 in_*.npy / out_*.npz）

用法（RDK X5）:
    python3 07_verify_accuracy_board.py --model xxx.bin --ref ./ref
    python3 07_verify_accuracy_board.py --model xxx.bin --ref ./ref --min-cos 0.99

判定：
    cosine >= min_cos           → PASS
    否则                        → WARN（量化误差偏大，考虑增加校准样本/换校准方法）
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

try:
    from hobot_dnn import pyeasy_dnn as dnn
except Exception as e:
    print("✘ 需要在 RDK X5 板端运行（hobot_dnn 不可用）: %s" % e)
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ref", default="./ref")
    ap.add_argument("--min-cos", type=float, default=0.99)
    ap.add_argument("--max-cases", type=int, default=5)
    args = ap.parse_args()

    meta_p = os.path.join(args.ref, "meta.json")
    if not os.path.exists(meta_p):
        print("✘ 找不到参考数据 %s（请先在宿主机跑 06_gen_reference.py）" % meta_p)
        return 1
    meta = json.load(open(meta_p))
    print("模型:", args.model)
    print("参考集: %s  输入模式=%s  尺寸=%s  样本数=%d" % (
        args.ref, meta["mode"], meta["size"], len(meta["items"])))

    models = dnn.load(args.model)
    print("子模型数:", len(models))

    ins = sorted(glob.glob(os.path.join(args.ref, "in_*.npy")))[: args.max_cases]
    all_cos = []
    for k, p in enumerate(ins):
        x = np.load(p)
        ref_p = os.path.join(args.ref, "out_%03d.npz" % k)
        if not os.path.exists(ref_p):
            continue
        ref = np.load(ref_p)
        outs = models[0].forward([x])
        print("\n-- 样本 %d  (%s -> %s)" % (k, os.path.basename(p), os.path.basename(ref_p)))
        for j, o in enumerate(outs):
            key = "out%d" % j
            if key not in ref:
                continue
            a = np.asarray(o.buffer).astype(np.float64).ravel()
            b = ref[key].astype(np.float64).ravel()
            if a.size != b.size:
                print("   %-6s 形状不一致: %d vs %d" % (key, a.size, b.size))
                continue
            cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
            d = np.abs(a - b)
            all_cos.append(cos)
            flag = "PASS" if cos >= args.min_cos else "WARN"
            print("   %-6s cos=%.6f  max|Δ|=%.5g  mean|Δ|=%.5g  [%s]" % (
                key, cos, d.max(), d.mean(), flag))

    if not all_cos:
        print("\n✘ 没有可比对的样本")
        return 1
    print("\n=== 汇总 ===")
    print("平均余弦相似度 = %.6f    最差 = %.6f    样本输出数 = %d" % (
        float(np.mean(all_cos)), float(np.min(all_cos)), len(all_cos)))
    print("阈值 min_cos=%.3f → %s" % (
        args.min_cos, "全部 PASS" if min(all_cos) >= args.min_cos else "存在 WARN，量化精度需关注"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
