import torch
import torch_npu

print("1. 探测 NPU:", torch.npu.is_available())
print("2. 准备创建 Tensor...")

# 强制触发 _npu_init()
try:
    x = torch.tensor([1.0, 2.0, 3.0]).to("npu")
except Exception as e:
    print(e)