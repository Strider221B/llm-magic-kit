
from helpers.config import Config
from helpers.constants import Constants
from models.vllm_based.base_model import BaseModel

class QWQ_32B(BaseModel):

    MODEL_PATH = f'{Config.BASE_PATH}/m/shelterw/qwen2.5/transformers/qwq-32b-preview-awq/1'

