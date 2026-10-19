#!/bin/bash
WS=/mnt/c/Users/20106/Documents/deepseek-harness/default-workspace
IMG=docker.jiaxin.site/openexplorer/ai_toolchain_ubuntu_20_x5_gpu:v1.2.8
D=/home/weydad/open_explorer/_project/yolov5

echo "=== 1. 目录现状（仓库是否克隆成功）==="
ls -la "$D" | head -14
echo "yolov5 仓库: $([ -d "$D/yolov5/.git" ] && echo 有 || echo 无)"
git -C "$D/yolov5" log -1 --format='  %h %cd' --date=short 2>/dev/null

echo
echo "=== 2. ONNX 算子与 I/O ==="
python3 - <<'PY'
import onnx
p = "/home/weydad/open_explorer/_project/yolov5/yolov5su.onnx"
m = onnx.load(p, load_external_data=False)
def sh(t): return [d.dim_value or d.dim_param for d in t.type.tensor_type.shape.dim]
print("opset:", [(o.domain or "ai.onnx", o.version) for o in m.opset_import])
print("输入:", [(i.name, sh(i)) for i in m.graph.input])
print("输出:", [(o.name, sh(o)) for o in m.graph.output])
ops = {}
for n in m.graph.node:
    ops[n.op_type] = ops.get(n.op_type, 0) + 1
print("节点数:", len(m.graph.node))
print("算子分布:", sorted(ops.items(), key=lambda kv: -kv[1]))
risky = [k for k in ops if any(x in k.lower() for x in ("grid", "deform", "topk", "nonmax", "scatternd"))]
print("需关注算子:", risky if risky else "无")
PY

echo
echo "=== 3. 写入转换配置 ==="
cp -v "$WS/yolov5su_config.yaml" "$D/"
cp -v "$WS/fcos_patch_toolchain.py" /home/weydad/open_explorer/toolchain_batch1_patch.py

echo
echo "=== 4. 启动脱离式转换（checker + makertbin）==="
docker rm -f yolo-build >/dev/null 2>&1
docker run -d --name yolo-build \
  -v /home/weydad/open_explorer:/open_explorer \
  -w /open_explorer/_project/yolov5 \
  --shm-size=8g \
  --entrypoint /usr/bin/bash \
  "$IMG" -lc '
    set -u
    python3 /open_explorer/toolchain_batch1_patch.py || exit 11
    echo "--- checker ---"
    hb_mapper checker --model-type onnx --march bayes-e --model yolov5su.onnx \
      --input-shape images 1x3x640x640 2>&1 | tail -14
    echo "--- makertbin START $(date +%T) ---"
    hb_mapper makertbin --config yolov5su_config.yaml --model-type onnx
    rc=$?
    echo "--- MAKERTBIN_EXIT=$rc $(date +%T) ---"
    ls -la model_output/*.bin 2>/dev/null
    exit $rc
  ' >/dev/null 2>&1

sleep 6
echo "容器: $(docker ps --filter name=yolo-build --format '{{.ID}}  {{.Status}}')"
echo
docker logs --tail 16 yolo-build 2>&1
