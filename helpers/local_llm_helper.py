import time

import pandas as pd
import transformers

from helpers.constants import Constants
from helpers.logger import Logger
from models.llm_model_wrapper import LLMModelWrapper
from models.prompts import Prompts
from models.model_factory import ModelFactory

#https://www.kaggle.com/code/sathyanarayanrao89/ai-mathematical-olympiad-phi2-basic/notebook

class LocalLLMHelper:

    _CUTOFF_TIME = time.time() + (4 * 60 + 45) * 60  # 4h45m timeout

    def __init__(self, 
                 model_wrapper: LLMModelWrapper):
        '''

        padding_side: for LLMs, the default of left is correct most of the times. Exception: fine tuning LLama with
        SFT where you might want to use right. https://github.com/huggingface/transformers/issues/34842#issuecomment-2528550342
        '''
        transformers.set_seed(42)
        self._model_wrapper = model_wrapper
        self._model = ModelFactory.get_model(model_wrapper)
        self._tokenizer = ModelFactory.get_tokenizer(self._model_wrapper, self._model)
    
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

            Logger.debug(f'********* question: {question}, answer: {answer}, \n')
            
            # Return prediction in required format
            return pd.DataFrame({'id': [id_], 'answer': [answer]})
            
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
