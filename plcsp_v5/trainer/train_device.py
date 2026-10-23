import os

# [local fix] single-GPU machine: expose GPU 0 before torch initializes CUDA.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from instance.generate_allsize_mofjsp_dataset import Logger
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.set_device(device)  # [local fix] CPU 上不调用
Logger.log("Training/device", f"device={device}, torch.cuda.set_device{device}", print_true=1)



# if __name__ == '__main__':

#     import torch
#     print("Available GPUs:", torch.cuda.device_count())   # 返回2，我只有01gpu能用，因为我的实例就是01gpu上边的 TODO 250506-修改为014567这6个GPU
