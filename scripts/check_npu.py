import torch
import torch_npu

def check_npu_devices():
    # 检查 NPU 是否可用
    if not torch.npu.is_available():
        print("未检测到可用的 NPU 环境。请检查驱动或 torch_npu 安装。")
        return

    device_count = torch.npu.device_count()
    print(f"当前环境检测到 {device_count} 张 NPU 显卡 (受环境变量影响)。\n")
    print("-" * 50)

    for i in range(device_count):
        device = torch.device(f"npu:{i}")
        
        # 获取设备名称
        name = torch.npu.get_device_name(i)
        
        # 获取显存信息 (转换为 GB)
        # 注意：这里获取的是 PyTorch 当前进程分配的显存，而不是全局 nvidia-smi/npu-smi 的物理显存
        memory_allocated = torch.npu.memory_allocated(i) / (1024 ** 3)
        memory_reserved = torch.npu.memory_reserved(i) / (1024 ** 3)
        
        # 也可以获取卡的总显存属性
        properties = torch.npu.get_device_properties(i)
        total_memory = properties.total_memory / (1024 ** 3)

        print(f"Device ID : {i}")
        print(f"Name      : {name}")
        print(f"Total Mem : {total_memory:.2f} GB")
        print(f"Allocated : {memory_allocated:.2f} GB")
        print("-" * 50)

if __name__ == "__main__":
    check_npu_devices()