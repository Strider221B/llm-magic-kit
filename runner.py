from datetime import datetime

from helpers.config import Config
from helpers.configs.env_configs import ConfigLocal
from helpers.local_llm_helper import LocalLLMHelper
from models.vllm_based.models import LLAMA_3_2_1B as model_wrapper

Config.override_env_defaults_with(ConfigLocal)

start_time = datetime.now()
llm_helper = LocalLLMHelper(model_wrapper)
print(f'\nFinal answer: {llm_helper.get_answer("What is 45*9?")}')
print(f'Elapsed time: {datetime.now() - start_time}')
