from helpers.config import Config
from helpers.constants import Constants
from helpers.logger import Logger
from models.llm_model_wrapper import LLMModelWrapper
from models.prompts import Prompts

class BaseModel(LLMModelWrapper):

    Config.MODEL_TYPE = Constants.VLLM

    TENSOR_PARALLEL_SIZE = 1 # means that the model itself is split across multiple GPUs.

    _MAX_TOKENS = 32768
    _MIN_CUMULATIVE_PROB_NUCLEUS_SAMPLING = 0.01
    _SKIP_SPECIAL_TOKENS = True
    _TEMPERATURE = 1.0

    @classmethod
    def get_model_response(cls, model, tokenizer, question: str) -> str:
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=cls._TEMPERATURE,                    # Controls randomness in generation: higher values (e.g., 1.0) produce more diverse output.
            min_p=cls._MIN_CUMULATIVE_PROB_NUCLEUS_SAMPLING, # Minimum cumulative probability for nucleus sampling, filtering out unlikely tokens.
            skip_special_tokens=cls._SKIP_SPECIAL_TOKENS,     
            max_tokens=cls._MAX_TOKENS,                      # Sets a very high limit for token generation to handle longer outputs.
        )

        try:
            prompt = Prompts.get_prompt(question)
            inputs = tokenizer.apply_chat_template(conversation=prompt,
                                                   tokenize=False,
                                                   add_generation_prompt=True)
            response = model.generate(prompts=[inputs],
                                      sampling_params=sampling_params)
            return response[0].outputs[0].text
            
        except Exception as e:
            Logger.exception(f"Generation error: {str(e)}")
            return f"Error occurred. Answer={Constants.DEFAULT_ANSWER}"
