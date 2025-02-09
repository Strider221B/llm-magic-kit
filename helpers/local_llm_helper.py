import time

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer

from models.constants import Constants
from models.llm_model_wrapper import LLMModelWrapper
from models.prompts import Prompts

#https://www.kaggle.com/code/sathyanarayanrao89/ai-mathematical-olympiad-phi2-basic/notebook

class LocalLLMHelper:

    _AUTO = 'auto'
    _CUDA = 'cuda'
    _DEVICE = _CUDA if torch.cuda.is_available() else 'cpu'
    _DEVICE_MAP = _AUTO if _DEVICE == _CUDA else None
    _LOW_CPU_MEM_USAGE = True
    _LOCAL_FILES_ONLY=True
    _TORCH_DTYPE = torch.float16 if _DEVICE == _CUDA else torch.float32
    _TRUST_REMOTE_CODE=True
    _CUTOFF_TIME = time.time() + (4 * 60 + 45) * 60  # 4h45m timeout

    def __init__(self, 
                 model_wrapper: LLMModelWrapper):
        '''

        padding_side: for LLMs, the default of left is correct most of the times. Exception: fine tuning LLama with
        SFT where you might want to use right. https://github.com/huggingface/transformers/issues/34842#issuecomment-2528550342
        '''
        self._model_wrapper = model_wrapper
        self._tokenizer = self._get_tokenizer(model_wrapper.MODEL_PATH, 
                                              model_wrapper.PAD_TOKEN, 
                                              model_wrapper.PADDING_SIDE)
        torch_dtype = model_wrapper.TORCH_DTYPE
        if not torch_dtype:
            torch_dtype = self._TORCH_DTYPE
        self._model = self._get_model(model_wrapper.MODEL_PATH, 
                                      torch_dtype)
    
    def predict(self, id_: pd.DataFrame, question: pd.DataFrame) -> pd.DataFrame:
        """Make a prediction.
        Args:
            id_: DataFrame containing the id column
            question: DataFrame containing the problem column
        Returns:
            DataFrame with id and answer columns
        """
        try:
            # Unpack values
            id_ = id_.item(0)
            question = question.item(0)
            
            # Get answer using our question-answering pipeline
            answer = self.get_answer(question)
            
            # Return prediction in required format
            return pd.DataFrame({'id': id_, 'answer': answer})
            
        except Exception as e:
            print(f"Prediction error for ID {id_}: {str(e)}")
            # Return default prediction on error
            return pd.DataFrame({'id': id_, 'answer': Constants.DEFAULT_ANSWER})

    def get_answer(self, 
                   question: str):
        """Complete pipeline with error handling and timeout."""
        try:
            # First check if we've exceeded the time limit
            if time.time() > self._CUTOFF_TIME:
                print("Time limit exceeded, returning default answer")
                return Constants.DEFAULT_ANSWER
                
            prompt = Prompts.get_prompt(question)
            response = self._model_wrapper.get_model_response(self._model, self._tokenizer, prompt)
            response = response.removeprefix(prompt)
            return Prompts.extract_answer(response)
        except Exception as e:
            print(f"Error in get_answer: {str(e)}")
            return Constants.DEFAULT_ANSWER

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
