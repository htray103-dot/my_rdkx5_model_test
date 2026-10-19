#!/bin/bash
# MonoOcc R50 主干 ONNX → X5 .bin（脱离式容器，含 checker + makertbin + 监控）
WS=/mnt/c/Users/20106/Documents/deepseek-harness/default-workspace
IMG=docker.jiaxin.site/openexplorer/ai_toolchain_ubuntu_20_x5_gpu:v1.2.8
R50_HOST=/home/weydad/open_explorer/_project/Mono_Occupancy/R50
R50_CONT=/open_explorer/_project/Mono_Occupancy/R50

echo "=== 0. 准备配置与补丁脚本 ==="
cp -v "$WS/monoocc_r50_trunk_config.yaml" "$R50_HOST/"
cp -v "$WS/fcos_patch_toolchain.py" /home/weydad/open_explorer/toolchain_batch1_patch.py
ls -la "$R50_HOST"/monoocc_r50_trunk.onnx "$R50_HOST"/monoocc_r50_trunk_config.yaml

echo
echo "=== 1. 启动脱离式容器（checker + makertbin）==="
docker rm -f mono-build >/dev/null 2>&1
docker run -d --name mono-build \
  -v /home/weydad/open_explorer:/open_explorer \
  -w "$R50_CONT" \
  --entrypoint /usr/bin/bash \
  "$IMG" -lc '
    set -u
    echo "--- 固定校准 batch=1（该模型输入固定 batch1）---"
    python3 /open_explorer/toolchain_batch1_patch.py || exit 11
    echo
    echo "--- hb_mapper checker ---"
    hb_mapper checker --model-type onnx --march bayes-e \
      --model monoocc_r50_trunk.onnx --input-shape images 1x3x384x1280 2>&1 | tail -18
    echo
    echo "--- makertbin START $(date +%T) ---"
    hb_mapper makertbin --config monoocc_r50_trunk_config.yaml --model-type onnx
    rc=$?
    echo "--- MAKERTBIN_EXIT=$rc $(date +%T) ---"
    ls -la model_output/*.bin 2>/dev/null
    exit $rc
  ' >/dev/null 2>&1

sleep 5
echo "容器: $(docker ps --filter name=mono-build --format '{{.ID}}  {{.Status}}')"
echo
echo "=== 2. 前几行日志 ==="
docker logs --tail 20 mono-build 2>&1
