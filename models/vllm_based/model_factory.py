import os

from vllm import LLM

from helpers.constants import Constants
from models.llm_model_wrapper import LLMModelWrapper

class ModelFactory:

    _GPU_MEMORY_UTILIZATION = 0.96
    _MAX_MODEL_LEN = 32768 #4096*10 Model context length.
    _TENSOR_PARALLEL_SIZE = 4 # means that the model itself is split across multiple GPUs.
    _TOKENIZERS_PARALLELISM = 'TOKENIZERS_PARALLELISM'
    _TRUST_REMOTE_CODE = True

    @classmethod
    def get_model(cls, 
                  model_wrapper: LLMModelWrapper):
        cls._initialize()
        return LLM(model_wrapper.MODEL_PATH,
                   max_model_len=cls._MAX_MODEL_LEN,         
                   trust_remote_code=cls._TRUST_REMOTE_CODE,     
                   tensor_parallel_size=cls._TENSOR_PARALLEL_SIZE,      
                   gpu_memory_utilization=cls._GPU_MEMORY_UTILIZATION)
    
    @classmethod
    def get_tokenizer(cls,
                      llm_model: LLM = None):
        
        return llm_model.get_tokenizer()
    
    @classmethod
    def _initialize(cls):
        os.environ[Constants.CUDA_VISIBLE_DEVICES] = "0,1,2,3"
        os.environ[cls._TOKENIZERS_PARALLELISM] = "false"
