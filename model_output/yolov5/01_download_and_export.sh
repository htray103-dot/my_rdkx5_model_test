#!/bin/bash
IMG=docker.jiaxin.site/openexplorer/ai_toolchain_ubuntu_20_x5_gpu:v1.2.8
D=/home/weydad/open_explorer/_project/yolov5

echo "=== 1. 容器内创建 _project/yolov5 并交回属主 ==="
docker run --rm -i -v /home/weydad/open_explorer:/open_explorer --entrypoint /usr/bin/bash "$IMG" -lc '
mkdir -p /open_explorer/_project/yolov5
chown -R 1000:1000 /open_explorer/_project/yolov5
ls -ld /open_explorer/_project/yolov5
' 2>&1 | tail -3

echo
echo "=== 2. 克隆官方 YOLOv5 仓库（浅克隆）==="
cd "$D" || exit 1
if [ -d "$D/yolov5/.git" ]; then
  echo "已存在仓库，跳过克隆"
else
  timeout 300 git clone --depth 1 https://github.com/ultralytics/yolov5.git "$D/yolov5" 2>&1 | tail -5
fi
echo "--- 仓库内容 ---"
ls "$D/yolov5" 2>/dev/null | head -12
echo "版本: $(git -C "$D/yolov5" log -1 --format='%h %cd' --date=short 2>/dev/null)"

echo
echo "=== 3. 下载权重并导出 ONNX（用 ultralytics，opset 11）==="
cd "$D" || exit 1
python3 - <<'PY'
import traceback
from ultralytics import YOLO

model = None
for name in ("yolov5s.pt", "yolov5su.pt"):
    try:
        model = YOLO(name)
        print("加载成功:", name)
        break
    except Exception as e:
        print("加载失败 %s: %s" % (name, str(e)[:200]))
if model is None:
    raise SystemExit("没有可用的 yolov5 权重")

try:
    out = model.export(format="onnx", opset=11, imgsz=640, simplify=False,
                       dynamic=False, batch=1, half=False)
    print("ONNX:", out)
except Exception:
    traceback.print_exc()
    raise SystemExit(1)
PY

echo
echo "=== 4. 产物 ==="
ls -la "$D"/*.pt "$D"/*.onnx 2>/dev/null
