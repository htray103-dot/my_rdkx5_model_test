#!/bin/bash
# ==============================================================================
#  在【工具链容器内】执行：ONNX 整形 → hb_mapper checker → makertbin(bayes-e)
#  用法：docker run ... <镜像> -c 'bash /open_explorer/_project/DETR/rtdetr/run-x5-convert.sh'
# ==============================================================================
set -uo pipefail
D=/open_explorer/_project/DETR/rtdetr
SRC="$D/rtdetrv2_r18vd_120e_coco_opset11.onnx"
RDK="$D/rtdetrv2_r18vd_120e_coco_rdk.onnx"
YAML="$D/rtdetrv2_r18vd_config.yaml"

echo "########## 0. 环境 ##########"
echo "python : $(python3 -V 2>&1)"
python3 -c "import onnx; print('onnx   :', onnx.__version__)" 2>&1
echo "hb_mapper: $(hb_mapper --version 2>&1 | tail -1)"

echo
echo "########## 1. ONNX 整形（固化 size 输入 / 固定形状 / opset<=11）##########"
if [ ! -f "$SRC" ]; then
  echo "✘ 找不到 $SRC"
  exit 1
fi
python3 "$D/make_rdk_onnx.py" "$SRC" "$RDK" --size 640 --ops-max 11 2>&1 | tail -25
[ -f "$RDK" ] || { echo "✘ 整形失败，没有产生 $RDK"; exit 1; }

echo
echo "########## 2. hb_mapper checker (march bayes-e) ##########"
hb_mapper checker --model-type onnx --march bayes-e --model "$RDK" \
  --input-shape images 1x3x640x640 2>&1 | tail -30
CHK=${PIPESTATUS[0]}
echo "checker 退出码=$CHK"

echo
echo "########## 3. hb_mapper makertbin ##########"
hb_mapper makertbin --config "$YAML" --model-type onnx 2>&1 | tail -50
MK=${PIPESTATUS[0]}
echo "makertbin 退出码=$MK"

echo
echo "########## 4. 产物 ##########"
ls -la "$D/model_output" 2>/dev/null
echo "-- 所有 .bin --"
find "$D" -maxdepth 3 -name '*.bin' -printf '%10s  %TY-%Tm-%Td %TH:%TM  %p\n' 2>/dev/null

echo
echo "########## 5. 把属主交回 weydad ##########"
chown -R 1000:1000 "$D" 2>/dev/null && echo "· 已 chown -R 1000:1000 $D"
exit $MK
