#!/bin/bash
# 【第1步·下载】RT-DETRv2-S (r18vd) 官方权重
# 目标: 拿到 rtdetrv2_r18vd_120e_coco_rerun_48.1.pth（约 77.4 MiB）
set -e
OUT="${1:-.}"
URL="https://github.com/lyuwenyu/storage/releases/download/v0.2/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth"
F="$OUT/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth"
mkdir -p "$OUT"
if [ -s "$F" ]; then echo "已存在: $F"; else
  echo "下载: $URL"
  curl -L --fail --connect-timeout 20 --retry 3 -o "$F" "$URL"
fi
ls -la "$F"
echo "sha256: $(sha256sum "$F" | awk '{print $1}')"
echo "（参考值 2ace52184b620204004509b72752ac7bfe64aadaf7fc1d076b18df8ab5a5c77e）"
echo "结构: {'ema': {'module': OrderedDict(538 张量, 20.25M 参数), 'updates': int}}"
