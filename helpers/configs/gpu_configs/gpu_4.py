import torch

class GPU_4:
    
    CUDA_VISIBLE_DEVICES = "0,1,2,3"
    TENSOR_PARALLEL_SIZE = 4 # means that the model itself is split across multiple GPUs.
    TORCH_DTYPE = torch.bfloat16