import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer

from helpers.config import Config
from helpers.constants import Constants
from models.llm_model_wrapper import LLMModelWrapper

class ModelFactory:

    _AUTO = 'auto'
    _DEVICE_MAP = _AUTO if Constants.DEVICE == Constants.CUDA else None
    _LOCAL_FILES_ONLY=True
    _LOW_CPU_MEM_USAGE = True
    _TRUST_REMOTE_CODE=True

    @classmethod
    def get_model(cls, 
                  model_wrapper: LLMModelWrapper):
        model = AutoModelForCausalLM.from_pretrained(
                model_wrapper.get_model_path(),
                device_map=cls._DEVICE_MAP,
                torch_dtype=Config.TORCH_DTYPE,
                trust_remote_code=cls._TRUST_REMOTE_CODE,
                # low_cpu_mem_usage should be used with device map: 
                # https://huggingface.co/docs/transformers/en/main_classes/model#transformers.PreTrainedModel.from_pretrained.low_cpu_mem_usage(bool,
                low_cpu_mem_usage=cls._LOW_CPU_MEM_USAGE,
                local_files_only=cls._LOCAL_FILES_ONLY
            )
            
        if Constants.DEVICE == Constants.CUDA:
            model.to(Constants.DEVICE)
        return model

    @classmethod
    def get_tokenizer(cls, 
                      model_wrapper: LLMModelWrapper) -> PreTrainedTokenizer:
        
        tokenizer = AutoTokenizer.from_pretrained(
                model_wrapper.get_model_path(),
                trust_remote_code=cls._TRUST_REMOTE_CODE,
                local_files_only=cls._LOCAL_FILES_ONLY,
                pad_token=model_wrapper.PAD_TOKEN,
                padding_side=model_wrapper.PADDING_SIDE
            )
        
        return tokenizer
