import random
import re

from models.constants import Constants

class Prompts:

    @staticmethod
    def get_prompt(question):
        """Format the question with better prompts that encourage step-by-step reasoning"""
        prompts = [
            "Please use chained reasoning to solve this step by step and provide the answer as Answer=[number]. Stop at answer, do not generate follow ups after generating answers.",
            "Please reflect and verify while reasoning, then provide the answer as Answer=[number]. Stop at answer, do not generate follow ups after generating answers.",
            "Solve this problem using concise and clear reasoning, providing the answer as Answer=[number]. Stop at answer, do not generate follow ups after generating answers.",
            "You are a helpful and reflective maths assistant. Please reason step by step and provide the answer as Answer=[number]. Stop at answer, do not generate follow ups after generating answers.",
            "You are the smartest maths expert. Please solve this precisely and provide the answer as Answer=[number]. Stop at answer, do not generate follow ups after generating answers."
        ]
        # Use random prompt or cycle through them
        prompt = random.choice(prompts)
        return f"{prompt}.\n Question: {question} \n Solution: Let's solve this step by step: "

    @staticmethod    
    def extract_answer(response):
        """More robust answer extraction"""
        try:
            # Try multiple answer formats
            if "Answer=" in response:
                answer = response.split("Answer=")[-1].strip().split()[0]
            elif "\\boxed{" in response:
                answer = response.split("\\boxed{")[-1].split("}")[0]
            elif "answer is" in response.lower():
                text_after = response.lower().split("answer is")[-1]
                numbers = re.findall(r'-?\d+\.?\d*', text_after)
                if numbers:
                    answer = numbers[0]
            else:
                numbers = re.findall(r'-?\d+\.?\d*', response)
                if numbers:
                    answer = numbers[-1]
                else:
                    print(f'Could not get answer from: {response}, returning default {Constants.DEFAULT_ANSWER}')
                    return Constants.DEFAULT_ANSWER
                    
            # Handle negative numbers and convert to int modulo 1000
            value = float(answer)
            if value < 0:
                value = abs(value)
            return int(value) % 1000
        except Exception as e:
            print(f'Got exception in extract_answer from {response}. Exception: {e}, returning default {Constants.DEFAULT_ANSWER}')
            return Constants.DEFAULT_ANSWER
        
if __name__ == '__main__':
    example_response = '''Solution: Let's solve this step by step: 
                            Step 1: Start with the number 1.
                            Step 2: Add another 1 to the number.
                            Answer: The answer is 2.

                            Follow-up exercises:
                            1. What is the answer if we change the question to "What is 2+2?"?
                            Solution: The solution would be the same as the original question, which is 4.
                            2. Can you explain why the answer to the question is always the same, regardless of the numbers used?
                            3. How would you explain the concept of addition to someone who has never heard of it before? Provide a step-by-step explanation. 

                            Solution to follow-up exercise 1:
                            The answer is still 4 because addition is a mathematical operation that combines two or more numbers to find their total or sum. In this case, we are adding 2 and 2 together, which gives us a total of 4. The order in which we add the numbers does not change the result, so whether we add 2 to 2 or 2 to 1 first, the answer will always be 4. This is known as the commutative property of addition, which states that changing the order of the addends does not affect the sum.
                            '''
    print(Prompts.extract_answer(example_response))