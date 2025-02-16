import re

from helpers.code_exec.python_repl import PythonREPL
from helpers.logger import Logger

class PythonFormatter:

    _REGEX_PYTHON_CODE = r'```python\s*(.*?)\s*```'
    _STD_PYTHON_IMPORTS = "import math\nimport numpy as np\nimport sympy as sp\n"

    @classmethod    
    def execute(cls, message: str) -> tuple[bool ,str]:
        success = False
        output = None
        python_code_list = cls._extract_python_code_list(message)
        for python_code in python_code_list:
            python_code = cls._process_python_code(python_code)
            try:
                success, output = PythonREPL()(python_code)
            except Exception as e:
                output = str(e)
                Logger.exception(f'python code output: {output}')
        return success, output

    @classmethod
    def _extract_python_code(cls, text):
        pattern = cls._REGEX_PYTHON_CODE
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            ans = "\n\n".join(matches)
            return ans
        return ""
    
    @classmethod
    def _extract_python_code_list(cls, text:str):
        pattern = cls._REGEX_PYTHON_CODE
        ans=[]
        matches = re.findall(pattern, text, re.DOTALL)
        for m in matches:
            ans.append(m)
        return ans

    @classmethod    
    def _process_python_code(cls, query):
        query = cls._STD_PYTHON_IMPORTS + query
        current_rows = query.strip().split("\n")
        new_rows = []
        for row in current_rows:
            new_rows.append(row)
        ans = "\n".join(new_rows)
        return ans
