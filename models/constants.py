import torch

class Constants:

    CUDA = 'cuda'
    DEFAULT_ANSWER = 210
    DEVICE = CUDA if torch.cuda.is_available() else 'cpu'
