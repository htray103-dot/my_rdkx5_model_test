#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 RT-DETRv2 导出的 ONNX 处理成地平线 hb_mapper(bayes-e / RDK X5) 可接受的形态：

  1. 把第二个输入 orig_target_sizes 固化成常量 [[S,S]]
     —— BPU 不支持 int64 输入，且多输入会迫使我们给校准数据配两路；
        上板时把 640x640 的框按实际比例缩放即可（后处理里做）。
  2. 把 images 输入固定为 1x3xSxS（去掉动态 batch 'N'）
  3. opset > 11 时降到 11（horizon_nn 只支持到 11）
  4. 可选：把 int64 输出转成 int32（--cast-int64）

用法: make_rdk_onnx.py <src.onnx> <dst.onnx> [--size 640] [--ops-max 11] [--cast-int64]
"""
import argparse
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto


def fix_input_size_of_images(g, size):
    for i in g.input:
        if i.name != "images":
            continue
        dims = i.type.tensor_type.shape.dim
        if len(dims) != 4:
            print("! images 维度不是 4，跳过固定形状")
            return
        for k, v in enumerate([1, 3, size, size]):
            dims[k].dim_param = ""
            dims[k].dim_value = v
        print("· images 固定为 1x3x%dx%d" % (size, size))


def bake_size_input(g, size):
    if not any(i.name == "orig_target_sizes" for i in g.input):
        print("· 没有 orig_target_sizes 输入，跳过固化")
        return
    for t in list(g.initializer):
        if t.name == "orig_target_sizes":
            g.initializer.remove(t)
    val = np.array([[size, size]], dtype=np.int64)
    g.initializer.append(numpy_helper.from_array(val, "orig_target_sizes"))
    for i in list(g.input):
        if i.name == "orig_target_sizes":
            g.input.remove(i)
    print("· orig_target_sizes 已固化为常量 [[%d, %d]]" % (size, size))


def fix_output_dims(g, size, num_queries):
    """把输出的符号维度换成静态值，便于 hb_mapper 做形状推导。"""
    want = {
        "labels": [1, num_queries],
        "boxes": [1, num_queries, 4],
        "scores": [1, num_queries],
    }
    for o in g.output:
        if o.name not in want:
            continue
        dims = o.type.tensor_type.shape.dim
        vals = want[o.name]
        if len(dims) != len(vals):
            print("! %s 维度数(%d)与预期(%d)不符，保持原样" % (o.name, len(dims), len(vals)))
            continue
        for k, v in enumerate(vals):
            dims[k].dim_param = ""
            dims[k].dim_value = v
        print("· 输出 %s 固定为 %s" % (o.name, "x".join(str(v) for v in vals)))


def cast_int64_outputs(g):
    changed = 0
    for out in list(g.output):
        if out.type.tensor_type.elem_type != TensorProto.INT64:
            continue
        name = out.name
        tmp = name + "_i64_tmp"
        for n in g.node:
            for k, o in enumerate(n.output):
                if o == name:
                    n.output[k] = tmp
        g.node.append(helper.make_node("Cast", [tmp], [name],
                                       to=TensorProto.INT32, name=name + "_cast_int32"))
        out.type.tensor_type.elem_type = TensorProto.INT32
        changed += 1
        print("· 输出 %s: int64 → int32" % name)
    if changed == 0:
        print("· 没有需要转换的 int64 输出")


def downgrade_opset(m, ops_max):
    vers = [o.version for o in m.opset_import if o.domain in ("", "ai.onnx")]
    cur = max(vers) if vers else ops_max
    if cur <= ops_max:
        print("· opset 已是 %d，无需降级" % cur)
        return m
    from onnx import version_converter
    try:
        m2 = version_converter.convert_version(m, ops_max)
        print("· opset %d → %d 成功" % (cur, ops_max))
        return m2
    except Exception as e:                                    # noqa: BLE001
        print("! opset 降级失败(%s)，保持 %d" % (e, cur))
        return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--num-queries", type=int, default=300)
    ap.add_argument("--ops-max", type=int, default=11)
    ap.add_argument("--cast-int64", action="store_true")
    ap.add_argument("--no-fix-out-dims", action="store_true")
    args = ap.parse_args()

    m = onnx.load(args.src)
    g = m.graph
    print("源模型 opset:", [(o.domain or "ai.onnx", o.version) for o in m.opset_import])
    print("源输入:", [i.name for i in g.input])

    bake_size_input(g, args.size)
    fix_input_size_of_images(g, args.size)
    if not args.no_fix_out_dims:
        fix_output_dims(g, args.size, args.num_queries)
    if args.cast_int64:
        cast_int64_outputs(g)

    m = downgrade_opset(m, args.ops_max)

    try:
        onnx.checker.check_model(m)
        print("· onnx.checker 通过")
    except Exception as e:                                    # noqa: BLE001
        print("! onnx.checker 未通过: %s" % e)
        print("  （先继续保存，交给 hb_mapper 判断）")

    onnx.save(m, args.dst)
    print("· 已保存:", args.dst)

    m2 = onnx.load(args.dst)
    print("结果 opset:", [(o.domain or "ai.onnx", o.version) for o in m2.opset_import])
    for i in m2.graph.input:
        print("  输入:", i.name,
              [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim],
              TensorProto.DataType.Name(i.type.tensor_type.elem_type))
    for o in m2.graph.output:
        print("  输出:", o.name,
              [d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim],
              TensorProto.DataType.Name(o.type.tensor_type.elem_type))
    return 0


if __name__ == "__main__":
    sys.exit(main())
