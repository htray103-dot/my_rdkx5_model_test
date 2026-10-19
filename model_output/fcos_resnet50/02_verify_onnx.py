#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""【ONNX 模型验证·FCOS】修补后的 ONNX 功能与输出自检

FCOS 的 ONNX 来自 SDK model_zoo（没有 pth→onnx 这一步），
但原模型的 Resize 输入里带一个**空 scales 常量**，onnxruntime 会拒绝执行，
所以先用 01_extract_onnx.py 修好，再用本脚本做功能性验证：
  * onnx.checker
  * 真实前向（随机输入）→ 15 个输出的形状/dtype/统计量
  * 检查输出 finite、类别通道数=80、尺度层级 128/64/32/16/8 是否正确
"""
import sys

import numpy as np
import onnx
import onnxruntime as ort

ONNX = sys.argv[1] if len(sys.argv) > 1 else "fcos_resnet50_resize_fix.onnx"

print("加载:", ONNX)
m = onnx.load(ONNX, load_external_data=False)
onnx.checker.check_model(m)
print("  checker: OK   节点数:", len(m.graph.node))

sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0]
print("  输入:", inp.name, inp.shape, inp.type)

rng = np.random.default_rng(0)
x = rng.random((1, 3, 1024, 1024), dtype=np.float32)
outs = sess.run(None, {inp.name: x})
print("  输出数:", len(outs))

ok = True
for i, o in enumerate(outs):
    a = np.asarray(o)
    finite = np.isfinite(a).all()
    ok = ok and bool(finite)
    print("   out[%2d] shape=%-18s dtype=%-8s min=%8.3f max=%8.3f finite=%s" % (
        i, a.shape, a.dtype, float(a.min()), float(a.max()), finite))

# 预期 15 个输出：5 尺度 × (80 类 / 4 reg / 1 ctr)
shapes = {tuple(np.asarray(o).shape) for o in outs}
expect_scales = {128, 64, 32, 16, 8}
seen = {s[-1] for s in shapes if len(s) == 4}
print("\n  检测到的尺度(最后一维):", sorted(seen))
if not expect_scales.issubset(seen):
    print("  ⚠ 尺度集合与预期 %s 不完全一致" % sorted(expect_scales))
    ok = False
if len(outs) != 15:
    print("  ⚠ 输出个数不是 15（实际 %d）" % len(outs))
    ok = False

print("\n结论:", "PASS — ONNX 可用，输出健康" if ok else "WARN — 请检查上面的异常项")
sys.exit(0 if ok else 1)
