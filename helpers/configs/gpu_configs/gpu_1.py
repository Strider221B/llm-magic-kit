import torch

class GPU_1:

    CUDA_VISIBLE_DEVICES = "0"
    TENSOR_PARALLEL_SIZE = 1 # means that the model itself is split across multiple GPUs.
    TORCH_DTYPE = torch.float16
