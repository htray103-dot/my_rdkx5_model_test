#!/bin/bash
# 主机端 .bin 验证：用工具链容器解析模型信息（不需要上板）
# 用法: bash verify_bin_host.sh <bin 的 WSL 路径>
IMG=docker.jiaxin.site/openexplorer/ai_toolchain_ubuntu_20_x5_gpu:v1.2.8
BIN="${1:-}"

if [ -z "$BIN" ] || [ ! -f "$BIN" ]; then
  echo "用法: bash $0 <bin 文件在 WSL 中的绝对路径>"
  echo "例:   bash $0 ~/open_explorer/_project/model_bin/fcos_resnet50_1024x1024_nv12.bin"
  exit 1
fi

# WSL 路径 → 容器内路径（只挂载了 open_explorer）
CONT="${BIN/\/home\/weydad\/open_explorer//open_explorer}"
if [ "$CONT" = "$BIN" ]; then
  echo "✘ 该文件不在 /home/weydad/open_explorer 下，容器看不到: $BIN"
  exit 1
fi

echo "=== 文件 ==="
ls -la "$BIN"
echo "sha256: $(sha256sum "$BIN" | awk '{print $1}')"

echo
echo "=== hb_model_info ==="
docker run --rm -i -v /home/weydad/open_explorer:/open_explorer --entrypoint /usr/bin/bash "$IMG" -lc "
  hb_model_info $CONT 2>&1 | grep -E 'BPU march|hb_mapper version|hbdk version|input_name|input_type_rt|input_type_train|input_layout_train|input_shape|norm_type|mean_value|scale_value|calibration_type|output_model_file_prefix'
" 2>&1 | sed 's/^[0-9-]* [0-9:,]* //' | sed 's/\x1b\[[0-9;]*m//g'

echo
echo "=== 结论 ==="
echo "若上面显示 BPU march: bayes-e 且输入输出形状符合预期，则模型可用于 RDK X5。"
echo "板端实测请把 .bin 拷到板子后运行同目录的 verify_*.py。"
