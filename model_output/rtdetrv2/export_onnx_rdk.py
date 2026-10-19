#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RT-DETRv2 → ONNX（地平线 RDK X5 友好版，自包含，不改动仓库文件）

与仓库自带 tools/export_onnx.py 的差别：
  1. 伪造 src 包，跳过 src/__init__.py 的重量级导入链
     （optim → misc → data → coco_dataset → faster_coco_eval）
     因此无需安装 faster-coco-eval / pycocotools / scipy
  2. 把第二输入 orig_target_sizes 固化成常量，直接导出【单输入、静态形状】模型
     —— BPU 不支持 int64 输入；上板时按实际比例缩放框即可
  3. opset 固定 11（horizon_nn 最高支持 11）

用法:
  python3 export_onnx_rdk.py -c <config.yml> -r <weights.pth> -o <out.onnx> [-s 640]
"""
import argparse
import os
import sys
import types

REPO_DEFAULT = "/home/weydad/open_explorer/_project/DETR/RT-DETR/rtdetrv2_pytorch"


def fake_src_package(repo):
    """让 `import src.core` 生效，但不执行 src/__init__.py 里的重量级导入。"""
    pkg = types.ModuleType("src")
    pkg.__path__ = [os.path.join(repo, "src")]
    sys.modules["src"] = pkg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", required=True)
    ap.add_argument("--resume", "-r", required=True)
    ap.add_argument("--output_file", "-o", default="model.onnx")
    ap.add_argument("--input_size", "-s", type=int, default=640)
    ap.add_argument("--repo", default=REPO_DEFAULT)
    ap.add_argument("--opset", type=int, default=11)
    ap.add_argument("--simplify", action="store_true")
    ap.add_argument("--no-sdpa-patch", action="store_true",
                    help="不替换 scaled_dot_product_attention（默认替换，否则 opset11 导不出）")
    ap.add_argument("--no-grid-sample-patch", action="store_true",
                    help="不替换 grid_sample（默认替换，否则 opset11 导不出）")
    args = ap.parse_args()

    repo = os.path.abspath(args.repo)
    sys.path.insert(0, repo)

    def load_yamlconfig():
        """优先常规导入；失败则伪造 src 包并只加载注册所需的最小模块集。"""
        try:
            from src.core import YAMLConfig as _Y
            print("· 常规导入 src.core 成功")
            return _Y
        except Exception as e:                                  # noqa: BLE001
            print("· 常规导入失败：%s" % e)
            print("  回退：伪造 src 包，跳过 src/__init__.py 的重量级注册导入")
            for k in [k for k in sys.modules if k == "src" or k.startswith("src.")]:
                del sys.modules[k]
            fake_src_package(repo)
            import importlib
            for mod in ("src.core", "src.nn", "src.zoo"):
                try:
                    importlib.import_module(mod)
                    print("  · 已导入", mod)
                except Exception as e2:                          # noqa: BLE001
                    print("  ! 导入 %s 失败: %s" % (mod, e2))
            from src.core import YAMLConfig as _Y
            return _Y

    YAMLConfig = load_yamlconfig()

    import torch
    import torch.nn as nn

    # 仓库脚本里的两行开关，只有 torch>=2.0 才有
    for flag in ("enable_mem_efficient_sdp", "enable_flash_sdp"):
        fn = getattr(torch.backends.cuda, flag, None)
        if callable(fn):
            try:
                fn(False)
            except Exception:
                pass

    # ---- 关键：torch 2.0 的 nn.MultiheadAttention 内部会调用融合算子
    #      aten::scaled_dot_product_attention，该算子无法导出到 opset 11。
    #      用等价的数学实现替换，让导出图里只剩 MatMul/Softmax 等普通算子。----
    if not args.no_sdpa_patch:
        import torch.nn.functional as F

        def _sdpa_math(query, key, value, attn_mask=None, dropout_p=0.0,
                       is_causal=False, scale=None, **kwargs):
            import math as _math
            sf = (1.0 / _math.sqrt(query.size(-1))) if scale is None else scale
            w = query @ key.transpose(-2, -1) * sf
            L, S = query.size(-2), key.size(-2)
            if is_causal:
                causal = torch.ones(L, S, dtype=torch.bool, device=query.device).tril()
                w = w.masked_fill(~causal, float("-inf"))
            if attn_mask is not None:
                if attn_mask.dtype == torch.bool:
                    w = w.masked_fill(~attn_mask, float("-inf"))
                else:
                    w = w + attn_mask
            w = torch.softmax(w, dim=-1)
            if dropout_p and dropout_p > 0.0:
                w = torch.dropout(w, dropout_p, train=True)
            return w @ value

        F.scaled_dot_product_attention = _sdpa_math
        print("· 已把 F.scaled_dot_product_attention 替换为数学实现")
    try:
        torch.backends.mha.set_fastpath_enabled(False)
        print("· 已关闭 MultiheadAttention fastpath")
    except Exception:
        pass

    # ---- 关键：F.grid_sample 对应的 ONNX GridSample 算子要求 opset>=16，
    #      而地平线 hb_mapper 只支持到 opset 11。
    #      RT-DETR 的 deformable attention 核心函数依赖 grid_sample，
    #      所以必须换成只用 opset11 已有算子的等价实现。----
    import torch.nn.functional as F

    _orig_grid_sample = F.grid_sample

    def _grid_sample_manual(input, grid, mode="bilinear", padding_mode="zeros",
                            align_corners=False):
        if mode != "bilinear" or padding_mode != "zeros":
            return _orig_grid_sample(input, grid, mode=mode,
                                     padding_mode=padding_mode,
                                     align_corners=align_corners)
        N, C, H, W = input.shape
        Hg, Wg = int(grid.shape[1]), int(grid.shape[2])
        x = grid[..., 0]
        y = grid[..., 1]
        if align_corners:
            ix = (x + 1.0) * (W - 1) / 2.0
            iy = (y + 1.0) * (H - 1) / 2.0
        else:
            ix = ((x + 1.0) * W - 1.0) / 2.0
            iy = ((y + 1.0) * H - 1.0) / 2.0

        x0 = torch.floor(ix)
        y0 = torch.floor(iy)
        x1 = x0 + 1.0
        y1 = y0 + 1.0
        wx1 = ix - x0
        wx0 = 1.0 - wx1
        wy1 = iy - y0
        wy0 = 1.0 - wy1

        flat = input.reshape(N, C, H * W)

        def corner(xs, ys):
            # zeros padding：越界的角点贡献 0（索引取夹紧值，乘上 in-bounds 掩码）
            inside = ((xs >= 0) & (xs <= W - 1) & (ys >= 0) & (ys <= H - 1)).to(input.dtype)
            xc = xs.clamp(0, W - 1).to(torch.int64)
            yc = ys.clamp(0, H - 1).to(torch.int64)
            idx = (yc * W + xc).reshape(N, 1, Hg * Wg).expand(N, C, Hg * Wg)
            v = torch.gather(flat, 2, idx).reshape(N, C, Hg, Wg)
            return v * inside.unsqueeze(1)

        return (corner(x0, y0) * (wx0 * wy0).unsqueeze(1)
                + corner(x1, y0) * (wx1 * wy0).unsqueeze(1)
                + corner(x0, y1) * (wx0 * wy1).unsqueeze(1)
                + corner(x1, y1) * (wx1 * wy1).unsqueeze(1))

    if not args.no_grid_sample_patch:
        # 自检：与原生实现比对（含越界采样点）
        _a = torch.randn(2, 4, 8, 8)
        _g = torch.rand(2, 5, 3, 2) * 2.6 - 1.3
        _ref = _orig_grid_sample(_a, _g, mode="bilinear",
                                 padding_mode="zeros", align_corners=False)
        _got = _grid_sample_manual(_a, _g)
        print("· grid_sample 替换自检：最大误差 = %.3e" % (_ref - _got).abs().max().item())
        F.grid_sample = _grid_sample_manual
        print("· 已把 F.grid_sample 替换为 opset11 可导出的等价实现")

    print("· 加载配置:", args.config)
    cfg = YAMLConfig(args.config)

    print("· 加载权重:", args.resume)
    ckpt = torch.load(args.resume, map_location="cpu")
    if isinstance(ckpt, dict) and "ema" in ckpt:
        state = ckpt["ema"]["module"]
        print("  (使用 ckpt['ema']['module']，共 %d 个张量)" % len(state))
    else:
        state = ckpt["model"]
        print("  (使用 ckpt['model'])")
    cfg.model.load_state_dict(state)

    s = args.input_size

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            # 固定尺寸常量：导出后即成为 ONNX 常量，不再是模型输入
            self.register_buffer("orig_target_sizes",
                                 torch.tensor([[s, s]], dtype=torch.int64))

        def forward(self, images):
            return self.postprocessor(self.model(images), self.orig_target_sizes)

    model = Model().eval()
    data = torch.rand(1, 3, s, s)
    with torch.no_grad():
        out = model(data)
    print("· 试跑输出:", [tuple(o.shape) for o in out])

    print("· 导出 ONNX (opset=%d, 单输入 1x3x%dx%d)" % (args.opset, s, s))
    torch.onnx.export(
        model,
        (data,),
        args.output_file,
        input_names=["images"],
        output_names=["labels", "boxes", "scores"],
        dynamic_axes=None,          # 静态形状，BPU 友好
        opset_version=args.opset,
        verbose=False,
        do_constant_folding=True,
    )

    import onnx
    m = onnx.load(args.output_file)
    onnx.checker.check_model(m)
    print("· onnx.checker 通过")
    print("  opset:", [(o.domain or "ai.onnx", o.version) for o in m.opset_import])
    for i in m.graph.input:
        print("  输入:", i.name,
              [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim])
    for o in m.graph.output:
        print("  输出:", o.name,
              [d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim])

    if args.simplify:
        try:
            import onnxsim
            ms, ok = onnxsim.simplify(args.output_file)
            if ok:
                onnx.save(ms, args.output_file)
                print("· onnxsim 简化完成")
        except Exception as e:                                  # noqa: BLE001
            print("· 跳过 onnxsim:", e)

    print("完成:", args.output_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
