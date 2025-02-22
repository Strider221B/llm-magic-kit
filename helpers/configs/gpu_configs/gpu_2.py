import torch

class GPU_2:

    CUDA_VISIBLE_DEVICES = "0,1"
    TENSOR_PARALLEL_SIZE = 2 # means that the model itself is split across multiple GPUs.
    TORCH_DTYPE = torch.float16
