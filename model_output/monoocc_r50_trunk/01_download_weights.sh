#!/bin/bash
# 【第1步·下载/准备】MonoOcc R50 权重 + 官方仓库
# MonoOcc-S.pth 来源：官方 Model Zoo（Google Drive）→ 已下载到 Windows 下载目录
set -e
OUT="${1:-/home/weydad/open_explorer/_project/Mono_Occupancy/R50}"
SRC_WIN="/mnt/c/Users/20106/Downloads/MonoOcc-S.pth"

echo "=== 1. 权重 ==="
if [ -s "$OUT/MonoOcc-S.pth" ]; then
  echo "已存在: $OUT/MonoOcc-S.pth"
elif [ -s "$SRC_WIN" ]; then
  cp -v "$SRC_WIN" "$OUT/MonoOcc-S.pth"
else
  echo "✘ 没找到 MonoOcc-S.pth"
  echo "  请从 https://github.com/ucaszyp/MonoOcc 的 Model Zoo（R50/SemanticKITTI）下载，"
  echo "  放到 $OUT/ 或 C:\\Users\\20106\\Downloads\\ 后重跑本脚本"
  exit 1
fi
ls -la "$OUT/MonoOcc-S.pth"
echo "sha256: $(sha256sum "$OUT/MonoOcc-S.pth" | awk '{print $1}')"
echo "结构: {'meta', 'state_dict'(513 张量, 59.03M 参数), 'optimizer'}"
echo "模块分布: img_backbone 318 / img_neck 4 / pts_bbox_head 191"

echo
echo "=== 2. 官方仓库（代码，用于核对结构）==="
REPO=/home/weydad/open_explorer/_project/Mono_Occupancy/MonoOcc
if [ -d "$REPO/.git" ]; then
  echo "已存在: $REPO  ($(git -C "$REPO" log -1 --format='%h %cd' --date=short 2>/dev/null))"
else
  echo "克隆 https://github.com/ucaszyp/MonoOcc.git"
  git clone --depth 1 https://github.com/ucaszyp/MonoOcc.git "$REPO"
fi
