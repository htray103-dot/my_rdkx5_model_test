import os
import numpy as np
import onnx
from onnx import numpy_helper, shape_inference

SRC = "/open_explorer/horizon_model_convert_sample/01_common/model_zoo/mapper/detection/fcos_resnet50/fcos_resnet50.onnx"
DST = "/open_explorer/horizon_model_convert_sample/04_detection/10_fcos_resnet50/mapper/fcos_resnet50_resize_fix.onnx"

print("加载:", SRC)
m = onnx.load(SRC)

# ---------- 1. 形状推断，确认 Resize 的放大倍率 ----------
mi = shape_inference.infer_shapes(m)
shape_of = {}
for coll in (mi.graph.value_info, mi.graph.input, mi.graph.output):
    for vi in coll:
        dims = vi.type.tensor_type.shape.dim
        shape_of[vi.name] = [d.dim_value if d.HasField("dim_value") else (d.dim_param or "?") for d in dims]


def show(name):
    return shape_of.get(name, "?")


for rn in [n for n in m.graph.node if n.op_type == "Resize"]:
    print("Resize %-12s  X=%s -> Y=%s" % (rn.name, show(rn.input[0]), show(rn.output[0])))

# ---------- 2. 把空 scales 常量替换为真实的 [1,1,2,2]，并去掉 sizes 输入 ----------
producer = {}
for n in m.graph.node:
    for o in n.output:
        producer[o] = n

fixed = []
for idx, n in enumerate(m.graph.node):
    if n.op_type != "Resize":
        continue
    if len(n.input) <= 2 or not n.input[2]:
        continue
    p = producer.get(n.input[2])
    if p is None or p.op_type != "Constant":
        print("跳过（scales 非常量）:", n.name)
        continue
    val = onnx.helper.get_attribute_value(p.attribute[0])
    if not ((len(val.raw_data) == 0 and len(val.float_data) == 0) or list(val.dims) == [0]):
        print("跳过（scales 非空）:", n.name)
        continue

    scales_name = "resize_scales_%d" % idx
    m.graph.initializer.append(
        numpy_helper.from_array(np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32), scales_name))

    old = list(n.input)
    # 保留 X, roi, 换成真实 scales；sizes 是拖尾可选输入，直接省略
    new_inputs = [old[0], old[1] if len(old) > 1 else "", scales_name]
    del n.input[:]
    n.input.extend(new_inputs)
    fixed.append((n.name, old, new_inputs))
    print("已修补 %-12s  %s -> %s" % (n.name, old, new_inputs))

print("修补数量:", len(fixed))
onnx.checker.check_model(m)
onnx.save(m, DST)
print("已保存:", DST, os.path.getsize(DST), "字节")

# ---------- 3. onnxruntime 真实前向验证 ----------
print()
print("=== onnxruntime 前向验证 ===")
import onnxruntime as ort
sess = ort.InferenceSession(DST, providers=["CPUExecutionProvider"])
x = np.random.rand(1, 3, 1024, 1024).astype(np.float32)
name = sess.get_inputs()[0].name
outs = sess.run(None, {name: x})
print("输入:", name, x.shape)
print("输出数:", len(outs))
for i, o in enumerate(outs):
    print("   out[%d] shape=%s" % (i, tuple(o.shape)))
print("FORWARD_OK")
