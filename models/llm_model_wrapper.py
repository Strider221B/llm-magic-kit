from abc import ABC

import torch

class LLMModelWrapper(ABC):

    MODEL_PATH = ''
    PAD_TOKEN = '</s>'
    PADDING_SIDE = 'left'
    TORCH_DTYPE = torch.float16

    @staticmethod
    def get_model_response(model, tokenizer, prompt: str) -> str:
        pass
