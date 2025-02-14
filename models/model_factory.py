from helpers.config import Config
from helpers.constants import Constants
from models.llm_model_wrapper import LLMModelWrapper
from models.transformer_based.model_factory import ModelFactory as transformer_factory
from models.vllm_based.model_factory import ModelFactory as vllm_factory

class ModelFactory:

    @classmethod
    def get_model(cls, 
                  model_wrapper: LLMModelWrapper):
        if Config.MODEL_TYPE == Constants.TRANSFORMERS:
            return transformer_factory.get_model(model_wrapper)
        if Config.MODEL_TYPE == Constants.VLLM:
            return vllm_factory.get_model(model_wrapper)

    @classmethod
    def get_tokenizer(cls, 
                      model_wrapper: LLMModelWrapper,
                      llm_model = None):
        if Config.MODEL_TYPE == Constants.TRANSFORMERS:
            return transformer_factory.get_tokenizer(model_wrapper)
        if Config.MODEL_TYPE == Constants.VLLM:
            return vllm_factory.get_tokenizer(llm_model)