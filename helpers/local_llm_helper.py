import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer

class LocalLLMHelper:

    _AUTO = 'auto'
    _CUDA = 'cuda'
    _DEVICE = _CUDA if torch.cuda.is_available() else 'cpu'
    _DEVICE_MAP = _AUTO if _DEVICE == _CUDA else None
    _LOW_CPU_MEM_USAGE = True
    _LOCAL_FILES_ONLY=True
    _TORCH_DTYPE = torch.float16 if _DEVICE == _CUDA else torch.float32
    _TRUST_REMOTE_CODE=True

    def __init__(self, 
                 model_path:str,
                 pad_token:str,
                 padding_side:str = 'left',
                 torch_dtype:torch.dtype=None):
        '''

        padding_side: for LLMs, the default of left is correct most of the times. Exception: fine tuning LLama with
        SFT where you might want to use right. https://github.com/huggingface/transformers/issues/34842#issuecomment-2528550342
        '''
        self._tokenizer = self._get_tokenizer(model_path, pad_token, padding_side)
        if not torch_dtype:
            torch_dtype = self._TORCH_DTYPE
        self._model = self._get_model(model_path, torch_dtype)

    @classmethod
    def _get_model(cls, 
                   model_path: str,
                   torch_dtype: torch.dtype):
        model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=cls._DEVICE_MAP,
                torch_dtype=torch_dtype,
                trust_remote_code=cls._TRUST_REMOTE_CODE,
                # low_cpu_mem_usage should be used with device map: 
                # https://huggingface.co/docs/transformers/en/main_classes/model#transformers.PreTrainedModel.from_pretrained.low_cpu_mem_usage(bool,
                low_cpu_mem_usage=cls._LOW_CPU_MEM_USAGE,
                local_files_only=cls._LOCAL_FILES_ONLY
            )
            
        if cls._DEVICE == cls._CUDA:
            model.to(cls._DEVICE)
        return model

    @classmethod
    def _get_tokenizer(cls, 
                       model_path: str,
                       pad_token: str,
                       padding_side: str) -> PreTrainedTokenizer:
        tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=cls._TRUST_REMOTE_CODE,
                local_files_only=cls._LOCAL_FILES_ONLY,
                pad_token=pad_token,
                padding_side=padding_side
            )
        
        return tokenizer