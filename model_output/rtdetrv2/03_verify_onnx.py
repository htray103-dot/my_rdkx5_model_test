#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RT-DETR 导出保真度验证：同一份权重下 PyTorch(补丁版) vs opset11 ONNX

为什么这样验证：
  * 之前对比的 opset16 ONNX 是 2024 年用 torch2.3.1 + 另一份权重导出的，不可比；
  * 这里用**同一个 .pth**，PyTorch 侧套用与导出时相同的补丁
    （SDPA→数学实现、grid_sample→手写双线性），直接和 ONNX 输出逐元素比。
  * 补丁本身的正确性另有独立自检：导出日志里 "grid_sample 替换自检：最大误差 = 1.192e-07"。
"""
import os
import sys

import numpy as np
import torch

REPO = "/home/weydad/open_explorer/_project/DETR/RT-DETR/rtdetrv2_pytorch"
CKPT = "/home/weydad/open_explorer/_project/DETR/rtdetr/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth"
ONNX = "/home/weydad/open_explorer/_project/DETR/rtdetr/rtdetrv2_r18vd_120e_coco_opset11.onnx"
IMG = "/home/weydad/open_explorer/_project/cal_data_dir/coco/coco128/images/train2017/000000000009.jpg"
SIZE = 640

sys.path.insert(0, REPO)
from src.core import YAMLConfig  # noqa: E402

import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402


# ---------- 与导出脚本完全一致的补丁 ----------
def _sdpa_math(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kw):
    import math
    sf = (1.0 / math.sqrt(query.size(-1))) if scale is None else scale
    w = query @ key.transpose(-2, -1) * sf
    return torch.softmax(w, dim=-1) @ value


def _grid_sample_manual(inp, grid, mode="bilinear", padding_mode="zeros", align_corners=False):
    N, C, H, W = inp.shape
    Hg, Wg = int(grid.shape[1]), int(grid.shape[2])
    x, y = grid[..., 0], grid[..., 1]
    ix = ((x + 1.0) * W - 1.0) / 2.0
    iy = ((y + 1.0) * H - 1.0) / 2.0
    x0, y0 = torch.floor(ix), torch.floor(iy)
    x1, y1 = x0 + 1.0, y0 + 1.0
    wx1, wy1 = ix - x0, iy - y0
    wx0, wy0 = 1.0 - wx1, 1.0 - wy1
    flat = inp.reshape(N, C, H * W)

    def corner(xs, ys):
        inside = ((xs >= 0) & (xs <= W - 1) & (ys >= 0) & (ys <= H - 1)).to(inp.dtype)
        xc = xs.clamp(0, W - 1).to(torch.int64)
        yc = ys.clamp(0, H - 1).to(torch.int64)
        idx = (yc * W + xc).reshape(N, 1, Hg * Wg).expand(N, C, Hg * Wg)
        return torch.gather(flat, 2, idx).reshape(N, C, Hg, Wg) * inside.unsqueeze(1)

    return (corner(x0, y0) * (wx0 * wy0).unsqueeze(1) + corner(x1, y0) * (wx1 * wy0).unsqueeze(1)
            + corner(x0, y1) * (wx0 * wy1).unsqueeze(1) + corner(x1, y1) * (wx1 * wy1).unsqueeze(1))


F.scaled_dot_product_attention = _sdpa_math
_orig_gs = F.grid_sample
F.grid_sample = _grid_sample_manual
try:
    torch.backends.mha.set_fastpath_enabled(False)
except Exception:
    pass  # torch 2.0 无此开关，忽略

# ---------- 建模 ----------
cfg = YAMLConfig(os.path.join(REPO, "configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml"))
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
state = ck["ema"]["module"] if "ema" in ck else ck["model"]
cfg.model.load_state_dict(state)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()
        self.register_buffer("sz", torch.tensor([[SIZE, SIZE]], dtype=torch.int64))

    def forward(self, images):
        return self.postprocessor(self.model(images), self.sz)


m = Model().eval()

from PIL import Image  # noqa: E402

im = Image.open(IMG).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
x = torch.from_numpy(np.transpose(np.asarray(im, dtype=np.float32) / 255.0, (2, 0, 1)))[None]

with torch.no_grad():
    t_out = [o.numpy() for o in m(x)]

import onnxruntime as ort  # noqa: E402

sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
o_out = sess.run(None, {sess.get_inputs()[0].name: x.numpy()})

print("图片:", os.path.basename(IMG))
print("PyTorch 输出:", [(o.shape, str(o.dtype)) for o in t_out])
print("ONNX    输出:", [(o.shape, str(o.dtype)) for o in o_out])
print()
names = ["labels", "boxes", "scores"]
for i, nm in enumerate(names):
    a = t_out[i].astype(np.float64).ravel()
    b = o_out[i].astype(np.float64).ravel()
    d = np.abs(a - b)
    eq = float((t_out[i] == o_out[i]).mean())
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    print("%-7s 完全相等比例=%.4f  max|Δ|=%.6g  mean|Δ|=%.6g  余弦=%.8f" % (
        nm, eq, d.max(), d.mean(), cos))

# top-10 框
sc_t = t_out[2].ravel()
top = np.argsort(-sc_t)[:10]
rel = np.abs(t_out[1].reshape(-1, 4)[top] - o_out[1].reshape(-1, 4)[top]) / SIZE
print("\ntop-10 框相对偏差(N/640): mean=%.6f  max=%.6f" % (rel.mean(), rel.max()))
print("结论: 若 labels 全等、boxes/scores 误差在 1e-4 量级，则 ONNX 导出保真。")
