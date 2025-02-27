from helpers.config import Config
from models.vllm_based.base_model import BaseModel

class DeelSeekR1Math_7B_Instruct(BaseModel):

    Config.MAX_MODEL_LEN = 4096 # Math model has a lower max model length
    MODEL_PATH = f'{Config.BASE_PATH}/deepseek-math/pytorch/deepseek-math-7b-instruct/1/'
