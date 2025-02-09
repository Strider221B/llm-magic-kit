from abc import ABC

import torch

class LLMModelWrapper(ABC):

    BASE_FOLDER = '/kaggle/input'
    MODEL_PATH = f'{BASE_FOLDER}/phi-2/pytorch/default/1'
    PAD_TOKEN = '</s>'
    PADDING_SIDE = 'left'
    TORCH_DTYPE = torch.float16

    @staticmethod
    def get_model_response(model, tokenizer, prompt: str) -> str:
        pass
