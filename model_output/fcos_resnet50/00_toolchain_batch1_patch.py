"""在工具链容器内临时修补 hmct 的校准批大小启发式。
FCOS 等固定 batch=1 的模型：hmct 会把模型改写成 batch=8 再前向，
但这类模型的输入 shape 是写死 1，onnxruntime 会直接拒绝，导致
makertbin 在校准首步静默退出（exit 255）。

这里把 get_batch_size 改为恒返回 1（工具链本身也有 batch=1 的回退路径）。
仅作用于本次运行的临时容器，不修改镜像。
"""
import sys

P = "/usr/local/lib/python3.10/dist-packages/hmct/quantizer/calibrater/activation/base.py"
OLD = "return 8 if input_support and not pyop_exist else 1"
NEW = "return 1  # patched by DSH: 固定 batch=1（该模型 batch>1 时 ORT 拒绝）"

src = open(P, encoding="utf-8").read()
if NEW in src:
    print("· 已经修补过，跳过")
    sys.exit(0)
if OLD not in src:
    print("✘ 未找到目标语句，未修改。文件片段：")
    for i, line in enumerate(src.splitlines()[118:135], start=119):
        print("%4d| %s" % (i, line))
    sys.exit(1)

open(P, "w", encoding="utf-8").write(src.replace(OLD, NEW))
print("· 已把 get_batch_size() 固定为 1")

# 回读确认
chk = open(P, encoding="utf-8").read()
print("· 校验:", "OK" if NEW in chk and OLD not in chk else "FAIL")
