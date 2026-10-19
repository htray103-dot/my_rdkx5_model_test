#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MonoOcc (R50) 主干分支导出为 ONNX —— 不依赖 mmcv / mmdet / mmdet3d 全栈

为什么只导主干：
  MonoOcc = ResNet50(C4) + FPN + MonoOccHead(BEV 128x128x16, 3 层 DeformCrossAttention
  + 2 层 DeformSelfAttention, 20 类)。头部的核心是 MSDeformableAttention3D
  （BEVFormer 系算法的自定义 CUDA 算子，ops/ 里是 .so），且要处理 3D 体素网格，
  BPU 无法承载。而主干（Conv+BN+ReLU+FPN）是标准算力密集型卷积网络，
  适合放 BPU。所以这里导出主干，头部留给 CPU/GPU。

实现要点：
  1. backbone 用 torchvision 的 resnet50（其参数命名与 checkpoint 里的
     img_backbone.* 完全一致：conv1/bn1/layer1..4），取 layer3 输出即 C4(1024ch, 1/16)。
  2. neck 就是 mmcv 的 FPN(num_outs=1)：
     lateral_convs.0 = Conv2d(1024,128,1)，fpn_convs.0 = Conv2d(128,128,3,pad=1)，
     且 act_cfg=None，所以两者之间没有任何激活。
  3. 权重从 MonoOcc-S.pth 的 state_dict['img_backbone.*'] / ['img_neck.*'] 严格加载。
  4. 导出 opset 11（地平线 hb_mapper 上限），静态形状。

用法：
  python3 monoocc_trunk_export.py --ckpt MonoOcc-S.pth -o monoocc_r50_trunk.onnx \
      --height 384 --width 1280
"""
import argparse
import os
import sys

import torch
import torch.nn as nn


class FPN1(nn.Module):
    """mmcv FPN(num_outs=1, out_channels=128, act_cfg=None) 的等价最小实现。"""

    def __init__(self):
        super().__init__()
        self.lateral_conv = nn.Conv2d(1024, 128, kernel_size=1)
        self.fpn_conv = nn.Conv2d(128, 128, kernel_size=3, padding=1)

    def forward(self, c4):
        return self.fpn_conv(self.lateral_conv(c4))


class MonoOccTrunk(nn.Module):
    """ResNet50 layer3 (C4) + FPN → [1, 128, H/16, W/16]"""

    def __init__(self, resnet, fpn):
        super().__init__()
        # 只跑到 layer3，避免把 layer4/fc 也导出
        self.stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3,
        )
        self.fpn = fpn

    def forward(self, x):
        return self.fpn(self.stem(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("-o", "--output", default="monoocc_r50_trunk.onnx")
    ap.add_argument("--height", type=int, default=384)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--opset", type=int, default=11)
    ap.add_argument("--fp16-check", action="store_true", help="额外打印 fp32 输出的统计量")
    args = ap.parse_args()

    print("torch:", torch.__version__)
    print("加载权重:", args.ckpt)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    # ---------- 1. backbone：torchvision ResNet50 ----------
    from torchvision.models import resnet50

    r50 = resnet50(weights=None)
    bb_state = {k[len("img_backbone."):]: v for k, v in sd.items() if k.startswith("img_backbone.")}
    print("checkpoint 里 backbone 张量数:", len(bb_state))
    missing, unexpected = r50.load_state_dict(bb_state, strict=False)
    print("  缺失(应为 fc.weight/fc.bias):", missing)
    print("  多余:", unexpected)
    if [m for m in missing if not m.startswith("fc.")]:
        print("✘ 有非 fc 的缺失键，权重与结构不匹配")
        return 1
    if unexpected:
        print("✘ 有多余键")
        return 1

    # ---------- 2. neck：FPN ----------
    fpn = FPN1()
    with torch.no_grad():
        fpn.lateral_conv.weight.copy_(sd["img_neck.lateral_convs.0.conv.weight"])
        fpn.lateral_conv.bias.copy_(sd["img_neck.lateral_convs.0.conv.bias"])
        fpn.fpn_conv.weight.copy_(sd["img_neck.fpn_convs.0.conv.weight"])
        fpn.fpn_conv.bias.copy_(sd["img_neck.fpn_convs.0.conv.bias"])
    print("FPN 权重已加载（1x1 1024→128, 3x3 128→128, 无激活）")

    model = MonoOccTrunk(r50, fpn).eval()

    # ---------- 3. 前向自检 ----------
    x = torch.randn(1, 3, args.height, args.width)
    with torch.no_grad():
        y = model(x)
    print("输入:", tuple(x.shape), " 输出:", tuple(y.shape))
    expect = (1, 128, args.height // 16, args.width // 16)
    if y.shape[-2:] != torch.Size(expect[-2:]) and y.shape[2:] != torch.Size(expect[2:]):
        print("⚠ 输出尺寸与预期 %s 不完全一致（可能因非 16 整数倍取整）" % (expect,))
    print("输出统计: min=%.4f max=%.4f mean=%.4f finite=%s" % (
        y.min().item(), y.max().item(), y.mean().item(), bool(torch.isfinite(y).all())))

    # ---------- 4. 导出 ----------
    print("导出 ONNX (opset=%d, 静态 1x3x%dx%d)" % (args.opset, args.height, args.width))
    torch.onnx.export(
        model,
        (x,),
        args.output,
        input_names=["images"],
        output_names=["feat"],
        dynamic_axes=None,
        opset_version=args.opset,
        do_constant_folding=True,
        verbose=False,
    )

    import onnx
    m = onnx.load(args.output)
    onnx.checker.check_model(m)
    print("onnx.checker 通过，opset:", [(o.domain or "ai.onnx", o.version) for o in m.opset_import])
    for i in m.graph.input:
        print("  输入:", i.name, [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim])
    for o in m.graph.output:
        print("  输出:", o.name, [d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim])
    print("节点数:", len(m.graph.node), " 初始化器:", len(m.graph.initializer))
    ops = {}
    for n in m.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    print("算子分布:", sorted(ops.items(), key=lambda kv: -kv[1]))
    print("完成:", args.output, os.path.getsize(args.output), "字节")
    return 0


if __name__ == "__main__":
    sys.exit(main())
