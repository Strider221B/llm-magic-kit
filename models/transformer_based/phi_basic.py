import torch

from helpers.config import Config
from helpers.constants import Constants
from helpers.gpu_helper import GPUHelper
from helpers.logger import Logger
from models.llm_model_wrapper import LLMModelWrapper
from models.prompts import Prompts

class Phi(LLMModelWrapper):

    Config.MODEL_TYPE = Constants.TRANSFORMERS
    MODEL_PATH = f'{Config.BASE_PATH}/phi-2/pytorch/default/1'

    @classmethod
    def get_model_response(cls, model, tokenizer, question: str) -> str:
        try:
            prompt = Prompts.get_prompt_with_question_only(question)
            inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=cls._get_max_token(),  # Increased for complex problems
                    temperature=0.7,  # More diverse solutions
                    num_beams=5,     # Increased beam search
                    top_p=0.95,      # Slightly higher nucleus sampling
                    top_k=50,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=True,  # Enable sampling
                    no_repeat_ngram_size=3
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            GPUHelper.clean_memory()
            return response.removeprefix(prompt)
            
        except Exception as e:
            Logger.exception(f"Generation error: {str(e)}")
            return f"Error occurred. Answer={Constants.DEFAULT_ANSWER}"
