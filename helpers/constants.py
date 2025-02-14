import torch

class Constants:

    CUDA = 'cuda'
    CUDA_VISIBLE_DEVICES = 'CUDA_VISIBLE_DEVICES'
    DEFAULT_ANSWER = 210
    DEVICE = CUDA if torch.cuda.is_available() else 'cpu'
    TRANSFORMERS = 'transformers'
    VLLM = 'vllm'
