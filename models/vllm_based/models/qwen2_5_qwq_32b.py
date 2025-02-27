from helpers.config import Config
from models.vllm_based.base_model import BaseModel

class Qwen2_5_QWQ_32B(BaseModel):

    MODEL_PATH = f'{Config.BASE_PATH}/qwen2.5/transformers/qwq-32b-preview-awq/1'

