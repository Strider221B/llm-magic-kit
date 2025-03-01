from helpers.config import Config
from models.vllm_based.base_model import BaseModel

class Deepseek_R1_Distill_LLAMA_8b(BaseModel):

    MODEL_PATH = f'{Config.BASE_PATH}/deepseek-r1/transformers/deepseek-r1-distill-llama-8b/2'

class Deepseek_R1_Distill_Qwen_1_5(BaseModel):

    MODEL_PATH = f'{Config.BASE_PATH}/deepseek-r1/transformers/deepseek-r1-distill-qwen-1.5b/2'

class DeepseekR1DistillQwen_14b(BaseModel):

    MODEL_PATH = f'{Config.BASE_PATH}/deepseek-r1/transformers/deepseek-r1-distill-qwen-14b/2'

class DeepseekR1DistillQwen_32b(BaseModel):

    MODEL_PATH = f'{Config.BASE_PATH}/deepseek-r1/transformers/deepseek-r1-distill-qwen-32b/2'

class DeelSeekR1Math_7B_Instruct(BaseModel):

    Config.MAX_MODEL_LEN = 4096 # Math model has a lower max model length
    MODEL_PATH = f'{Config.BASE_PATH}/deepseek-math/pytorch/deepseek-math-7b-instruct/1/'

class QWEN_2_5_1_5_Instruct(BaseModel):

    MODEL_PATH = f'{Config.BASE_PATH}/m/qwen-lm/qwen2.5/transformers/1.5b-instruct/1'

class QWEN_2_5_Math_7B_Instruct(BaseModel):
    MODEL_PATH = f'{Config.BASE_PATH}/qwen2.5-math/transformers/7b-instruct/1'

class Qwen2_5_QWQ_32B(BaseModel):

    MODEL_PATH = f'{Config.BASE_PATH}/qwen2.5/transformers/qwq-32b-preview-awq/1'
