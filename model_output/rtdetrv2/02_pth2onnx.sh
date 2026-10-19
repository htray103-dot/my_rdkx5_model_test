#!/usr/bin/env bash
# ==============================================================================
#  RT-DETRv2-S 权重转换： .pth  ->  .onnx
#  文件位置: /home/weydad/open_explorer/_project/DETR/rtdetr/pth2onnx.sh
#
#  用法（会自动识别所在环境）:
#     容器内(推荐):  docker run ... <镜像> -c 'bash /open_explorer/_project/DETR/rtdetr/pth2onnx.sh'
#     WSL 宿主机:    bash /home/weydad/open_explorer/_project/DETR/rtdetr/pth2onnx.sh
#
#  注意: 本镜像的 ENTRYPOINT 是 /bin/bash，所以 `docker run <镜像> python3 xxx`
#        会被解释成 `bash python3 xxx` 而报 "cannot execute binary file"。
#        正确写法是 `docker run <镜像> -c '命令'`，或用 --entrypoint 覆盖。
#
#  可选环境变量:
#     ONNX=/path/out.onnx   输出路径（默认与权重同目录）
#     INPUT_SIZE=640        输入分辨率
#     NO_SIMPLIFY=1         跳过 onnxsim 图简化
#     NO_INSTALL=1          缺依赖时只报错、不自动 pip 安装
#     OWNER=1000:1000       以 root 运行时，产物 chown 给该属主
# ==============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKPT="${CKPT:-$SCRIPT_DIR/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth}"
ONNX="${ONNX:-$SCRIPT_DIR/rtdetrv2_r18vd_120e_coco.onnx}"
INPUT_SIZE="${INPUT_SIZE:-640}"

# ---------- 1. 定位仓库（容器内 / 宿主机） ----------
if [ -d /open_explorer/_project/DETR/RT-DETR/rtdetrv2_pytorch ]; then
  REPO=/open_explorer/_project/DETR/RT-DETR/rtdetrv2_pytorch
  WHERE="容器内"
elif [ -d /home/weydad/open_explorer/_project/DETR/RT-DETR/rtdetrv2_pytorch ]; then
  REPO=/home/weydad/open_explorer/_project/DETR/RT-DETR/rtdetrv2_pytorch
  WHERE="WSL 宿主机"
else
  echo "✘ 找不到 rtdetrv2_pytorch 仓库目录" >&2
  exit 1
fi
CONFIG="$REPO/configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml"

echo "=============================================================="
echo " RT-DETRv2-S  .pth → .onnx"
echo "--------------------------------------------------------------"
echo " 运行环境 : $WHERE   (uid=$(id -u))"
echo " 仓库     : $REPO"
echo " 配置     : $CONFIG"
echo " 权重     : $CKPT"
echo " 输出     : $ONNX"
echo " 分辨率   : ${INPUT_SIZE}x${INPUT_SIZE}"
echo "=============================================================="

# ---------- 2. 前置检查 ----------
for f in "$CKPT" "$CONFIG" "$REPO/tools/export_onnx.py"; do
  [ -f "$f" ] || { echo "✘ 缺少必要文件: $f" >&2; exit 1; }
done

# ---------- 3. 依赖检查 / 按需安装 ----------
MISSING=()
for m in torch torchvision yaml onnx; do
  python3 -c "import $m" >/dev/null 2>&1 || MISSING+=("$m")
done
# RT-DETR 的 src.core 会间接 import tensorboard，缺了直接 ModuleNotFoundError
python3 -c "import tensorboard" >/dev/null 2>&1 || MISSING+=("tensorboard")

if [ "${#MISSING[@]}" -gt 0 ]; then
  echo
  echo "⚠ 缺少依赖: ${MISSING[*]}"
  if [ "${NO_INSTALL:-0}" = "1" ]; then
    echo "  已设 NO_INSTALL=1，请自行安装后重跑："
    echo "    python3 -m pip install --user ${MISSING[*]}"
    exit 1
  fi
  echo "  → python3 -m pip install --user ${MISSING[*]}"
  python3 -m pip install --user "${MISSING[@]}" || { echo "✘ 依赖安装失败" >&2; exit 1; }
fi

SIMPLIFY_FLAG="--simplify"
if [ "${NO_SIMPLIFY:-0}" = "1" ]; then
  SIMPLIFY_FLAG=""
  echo "· 按要求跳过 onnxsim 简化"
elif ! python3 -c "import onnxsim" >/dev/null 2>&1; then
  echo "· 未装 onnxsim，尝试安装…"
  if [ "${NO_INSTALL:-0}" != "1" ] && python3 -m pip install --user onnxsim >/dev/null 2>&1; then
    echo "· onnxsim 安装成功"
  else
    echo "· onnxsim 不可用，本次不做图简化"
    SIMPLIFY_FLAG=""
  fi
fi

# ---------- 4. 导出 ----------
MARKER="$(mktemp)"
echo
echo ">>> 导出开始 $(date '+%F %T')"
cd "$REPO" || exit 1
PYTHONPATH="$REPO" python3 tools/export_onnx.py \
  -c "$CONFIG" \
  -r "$CKPT" \
  -o "$ONNX" \
  -s "$INPUT_SIZE" \
  --check $SIMPLIFY_FLAG
RC=$?
echo ">>> 导出结束 $(date '+%F %T')  退出码=$RC"

# 判成功必须同时满足：进程退出码为 0，且产物确实比"本次开始"更新。
# 否则目录里已有的旧 .onnx 会被误判成刚生成的结果。
if [ "$RC" -ne 0 ]; then
  rm -f "$MARKER"
  echo "✘ 导出进程失败（退出码 $RC）" >&2
  echo "  提示：若报 'no attribute enable_mem_efficient_sdp'，说明本环境 torch < 2.0，" >&2
  echo "        请在宿主机(或 torch>=2.0 的环境)执行本脚本。" >&2
  exit "$RC"
fi
if [ ! -f "$ONNX" ] || [ "$ONNX" -ot "$MARKER" ]; then
  rm -f "$MARKER"
  echo "✘ 没有生成新的 $ONNX（目录里若有同名旧文件，不算成功）" >&2
  exit 1
fi
rm -f "$MARKER"

# ---------- 5. 属主修正（容器内 root 产出时，交给 weydad） ----------
if [ "$(id -u)" = "0" ]; then
  OWNER="${OWNER:-1000:1000}"
  chown "$OWNER" "$ONNX" 2>/dev/null && echo "· 已把产物属主改为 $OWNER"
fi

# ---------- 6. 结果校验 ----------
echo
echo "✔ 生成成功"
ls -la "$ONNX"
sha256sum "$ONNX"
python3 - "$ONNX" <<'PY'
import sys
p = sys.argv[1]
try:
    import onnx
    m = onnx.load(p)
    onnx.checker.check_model(m)
    def shape(t):
        return [d.dim_value or d.dim_param for d in t.type.tensor_type.shape.dim]
    print('onnx.checker : 通过')
    print('输入         :', [(i.name, t) for i, t in [(i.name, shape(i)) for i in m.graph.input]])
    print('输入         :', [(i.name, shape(i)) for i in m.graph.input])
    print('输出         :', [(o.name, shape(o)) for o in m.graph.output])
    print('opset        :', m.opset_import[0].version, ' 节点数:', len(m.graph.node))
except Exception as e:
    print('onnx 校验未通过:', e)
PY
