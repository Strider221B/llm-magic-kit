import time
from abc import ABC

from helpers.config import Config

class LLMModelWrapper(ABC):

    PAD_TOKEN = '</s>'
    PADDING_SIDE = 'left'

    _MODEL_PATH = ''

    @classmethod
    def get_model_path(cls) -> str:
        return f'{Config.BASE_PATH}{cls._MODEL_PATH}'

    @staticmethod
    def get_model_response(model, tokenizer, prompt: str) -> str:
        pass

    @staticmethod
    def _get_max_token() -> int:
        '''
        Reduces max token in proportion to remainting time
        '''
        max_tokens = Config.MAX_TOKENS
        return int((max_tokens // 4) + 0.75 * max_tokens * ((Config.CUTOFF_TIME - time.time()) / Config.DURATION))
