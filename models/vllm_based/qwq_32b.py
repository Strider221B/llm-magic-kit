
from helpers.config import Config
from helpers.constants import Constants
from models.llm_model_wrapper import LLMModelWrapper

class QWQ_32B(LLMModelWrapper):

    MODEL_PATH = f'{Config.BASE_PATH}/m/shelterw/qwen2.5/transformers/qwq-32b-preview-awq/1'

