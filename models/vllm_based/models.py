from helpers.config import Config
from models.vllm_based.base_model import BaseModel

class Deepseek_R1_Distill_LLAMA_8b(BaseModel):

    _MODEL_PATH = 'deepseek-r1/transformers/deepseek-r1-distill-llama-8b/2'

class Deepseek_R1_Distill_Qwen_1_5(BaseModel):

    _MODEL_PATH = 'deepseek-r1/transformers/deepseek-r1-distill-qwen-1.5b/2'

class DeepseekR1DistillQwen_14b(BaseModel):

    _MODEL_PATH = 'deepseek-r1/transformers/deepseek-r1-distill-qwen-14b/2'

class DeepseekR1DistillQwen_32b(BaseModel):

    _MODEL_PATH = 'deepseek-r1/transformers/deepseek-r1-distill-qwen-32b/2'

class DeelSeekR1Math_7B_Instruct(BaseModel):

    Config.MAX_MODEL_LEN = 4096 # Math model has a lower max model length
    _MODEL_PATH = 'deepseek-math/pytorch/deepseek-math-7b-instruct/1/'

class LLAMA_3_2_1B(BaseModel):
    # To get LLAMA working locally with vLLM, had to add
    #   "chat_template": "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}",
    # to:  ~/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08/tokenizer_config.json
    _MODEL_PATH = 'meta-llama/Llama-3.2-1B'

class QWEN_2_5_1_5_Instruct(BaseModel):
    _MODEL_PATH = 'qwen2.5/transformers/1.5b-instruct/1'

class QWEN_2_5_Math_7B_Instruct(BaseModel):
    _MODEL_PATH = 'qwen2.5-math/transformers/7b-instruct/1'

class Qwen2_5_QWQ_32B(BaseModel):

    _MODEL_PATH = 'qwen2.5/transformers/qwq-32b-preview-awq/1'
