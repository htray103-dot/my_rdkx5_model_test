#!/bin/bash
WS=/mnt/c/Users/20106/Documents/deepseek-harness/default-workspace
IMG=docker.jiaxin.site/openexplorer/ai_toolchain_ubuntu_20_x5_gpu:v1.2.8
MAPPER=/open_explorer/horizon_model_convert_sample/04_detection/10_fcos_resnet50/mapper

cp -v "$WS/fcos_patch_toolchain.py" /home/weydad/open_explorer/fcos_patch_toolchain.py

echo "=== 启动（脱离式）：先补工具链，再跑完整 PTQ ==="
docker rm -f fcos-build2 >/dev/null 2>&1
docker run -d --name fcos-build2 \
  -v /home/weydad/open_explorer:/open_explorer \
  -w "$MAPPER" \
  --entrypoint /usr/bin/bash \
  "$IMG" -lc '
    set -u
    echo "===== 步骤1: 修补工具链批大小 ====="
    python3 /open_explorer/fcos_patch_toolchain.py || exit 11
    echo
    echo "===== 步骤2: makertbin 开始 $(date +%F\ %T) ====="
    hb_mapper makertbin --config fcos_resnet50_config_fixed.yaml --model-type onnx
    rc=$?
    echo "===== MAKERTBIN_EXIT=$rc  $(date +%F\ %T) ====="
    ls -la model_output/*.bin 2>/dev/null
    exit $rc
  ' >/dev/null 2>&1

sleep 5
echo "容器: $(docker ps --filter name=fcos-build2 --format '{{.ID}}  {{.Status}}')"
echo
echo "=== 日志 ==="
docker logs --tail 12 fcos-build2 2>&1
