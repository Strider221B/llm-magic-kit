from helpers.config import Config
from models.vllm_based.base_model import BaseModel

class QWEN_2_5_1_5_Instruct(BaseModel):

    MODEL_PATH = f'{Config.BASE_PATH}/m/qwen-lm/qwen2.5/transformers/1.5b-instruct/1'
