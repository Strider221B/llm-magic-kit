from helpers.config import Config
from helpers.constants import Constants
from models.vllm_based.base_model import BaseModel

class Qwen2_5_QWQ_32B(BaseModel):

    TENSOR_PARALLEL_SIZE = 4 # means that the model itself is split across multiple GPUs.
    CUDA_VISIBLE_DEVICES = "0,1,2,3"

    MODEL_PATH = f'{Config.BASE_PATH}/qwen2.5/transformers/qwq-32b-preview-awq/1'

