from helpers.config import Config
from models.vllm_based.base_model import BaseModel

class Deepseek_R1_Distill_LLAMA_8b(BaseModel):

    MODEL_PATH = f'{Config.BASE_PATH}/deepseek-r1/transformers/deepseek-r1-distill-llama-8b/2'
