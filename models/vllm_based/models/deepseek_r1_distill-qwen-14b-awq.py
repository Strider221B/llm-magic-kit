from helpers.config import Config
from models.vllm_based.base_model import BaseModel

class DeepseekR1DistillQwen_14b_awq(BaseModel):

    MODEL_PATH = f'{Config.BASE_PATH}/deepseek-r1/transformers/deepseek-r1-distill-qwen-1.5b/2'
