from abc import ABC

import torch

class LLMModelWrapper(ABC):

    CUDA_VISIBLE_DEVICES = "0"
    MODEL_PATH = ''
    PAD_TOKEN = '</s>'
    PADDING_SIDE = 'left'
    TORCH_DTYPE = torch.float16

    @staticmethod
    def get_model_response(model, tokenizer, prompt: str) -> str:
        pass
