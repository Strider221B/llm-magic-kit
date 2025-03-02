import random
import re
from collections import Counter
from typing import Dict, List

from helpers.constants import Constants
from helpers.logger import Logger

class Prompts:

    _REGEX_NUMBERS = r'-?\d+\.?\d*'
    _VALID_ANS_MAX = 1000
    _VALID_ANS_MIN = 0

    _SYSTEM_PROMPTS = [('You are the smartest maths expert. Given this problem please use chained reasoning to solve this step by step. '
                        'Also, generate Python code to get to the solution using sympy and numpy library if it helps. '
                        'Provide the answer as Answer=[number]. Stop at answer, do not generate follow ups after generating answers. ')]

    @classmethod
    def get_prompt(cls, question: str) -> List[Dict[str, str]]:
        prompt = [
            {"role": "system", "content": cls._get_random_system_prompt()},
            {"role": "user", "content": question}
        ]
        return prompt

    @classmethod
    def get_prompt_with_question_only(cls, question) -> str:
        '''Format the question with better prompts that encourage step-by-step reasoning'''
        # Use random prompt or cycle through them
        prompt = cls._get_random_system_prompt()
        return f'{prompt}.\n Question: {question} \n Solution: Let us solve this step by step: '

    @classmethod    
    def extract_answer(cls, response):
        '''More robust answer extraction'''
        try:
            # Try multiple answer formats
            if 'Answer=' in response:
                answer = response.split('Answer=')[-1].strip().split()[0]
            elif '\\boxed{' in response:
                answer = response.split('\\boxed{')[-1].split('}')[0]
            elif 'answer is' in response.lower():
                text_after = response.lower().split('answer is')[-1]
                numbers = re.findall(cls._REGEX_NUMBERS, text_after)
                if numbers:
                    answer = numbers[0]
            else:
                numbers = re.findall(cls._REGEX_NUMBERS, response)
                if numbers:
                    answer = numbers[-1]
                else:
                    Logger.error(f'Could not get answer from: {response}, returning default {Constants.DEFAULT_ANSWER}')
                    return Constants.DEFAULT_ANSWER
                    
            # Handle negative numbers and convert to int modulo 1000
            answer = cls._extract_number_from(answer)
            value = float(answer)
            if value < 0:
                value = abs(value)
            return int(value) % 1000
        except Exception as e:
            Logger.exception(f'Got exception in extract_answer from {response}. Exception: {e}, returning default {Constants.DEFAULT_ANSWER}')
            return Constants.DEFAULT_ANSWER
    
    @classmethod
    def select_most_common_answer(cls, answers):
        valid_answers = []
        for answer in answers:
            try:
                int_answer = int(answer)
                if float(answer).is_integer():
                    if cls._VALID_ANS_MIN <= int_answer <= cls._VALID_ANS_MAX:
                        valid_answers.append(int_answer)
            except:
                pass
        if not valid_answers:
            Logger.error(f'No valid answer found in {answers}, returning default')
            return Constants.DEFAULT_ANSWER
        _, answer = sorted([(v,k) for k,v in Counter(valid_answers).items()], reverse=True)[0]
        return answer%1000
    
    @classmethod
    def extract_answer_from_python_output(cls, is_success: bool, output: str):
        result = []
        if is_success:
            matches = re.findall(cls._REGEX_NUMBERS, output)
            if matches:
                for match in matches:
                    result.append(int(match)%1000)
        return result
    
    @classmethod
    def _get_random_system_prompt(cls) -> str:
        return random.choice(cls._SYSTEM_PROMPTS)
    
    @staticmethod
    def _extract_number_from(string):
        return re.sub(r'\D', '', string)
        
if __name__ == '__main__':
    example_response = '''Solution: Let's solve this step by step: 
                            Step 1: Start with the number 1.
                            Step 2: Add another 1 to the number.
                            Answer: The answer is 2.

                            Follow-up exercises:
                            1. What is the answer if we change the question to 'What is 2+2?'?
                            Solution: The solution would be the same as the original question, which is 4.
                            2. Can you explain why the answer to the question is always the same, regardless of the numbers used?
                            3. How would you explain the concept of addition to someone who has never heard of it before? Provide a step-by-step explanation. 

                            Solution to follow-up exercise 1:
                            The answer is still 4 because addition is a mathematical operation that combines two or more numbers to find their total or sum. In this case, we are adding 2 and 2 together, which gives us a total of 4. The order in which we add the numbers does not change the result, so whether we add 2 to 2 or 2 to 1 first, the answer will always be 4. This is known as the commutative property of addition, which states that changing the order of the addends does not affect the sum.
                            '''
    print(Prompts.extract_answer(example_response))