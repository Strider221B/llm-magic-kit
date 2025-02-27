import os

from helpers.config import Config
from models.vllm_based.base_model import BaseModel

class ModelFactory:

    _GPU_MEMORY_UTILIZATION = 0.96
    _TOKENIZERS_PARALLELISM = 'TOKENIZERS_PARALLELISM'
    _TRUST_REMOTE_CODE = True
    _VLLM_CUDA_MULTIPROCESSING_METHOD = 'VLLM_WORKER_MULTIPROC_METHOD'
    _MULTI_PROCESSING_SPAWN = "spawn"

    @classmethod
    def get_model(cls, 
                  model_wrapper: BaseModel):
        cls._initialize(model_wrapper)
        from vllm import LLM
        return LLM(model_wrapper.MODEL_PATH,
                   max_model_len=Config.MAX_MODEL_LEN,         
                   trust_remote_code=cls._TRUST_REMOTE_CODE,     
                   tensor_parallel_size=Config.TENSOR_PARALLEL_SIZE,      
                   gpu_memory_utilization=cls._GPU_MEMORY_UTILIZATION,
                   dtype=Config.TORCH_DTYPE)
    
    @classmethod
    def get_tokenizer(cls,
                      llm_model = None):
        return llm_model.get_tokenizer()
    
    @classmethod
    def _initialize(cls, model: BaseModel):
        os.environ[cls._TOKENIZERS_PARALLELISM] = "false"
        os.environ[cls._VLLM_CUDA_MULTIPROCESSING_METHOD] = cls._MULTI_PROCESSING_SPAWN
