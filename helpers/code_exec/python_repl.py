import os
import tempfile
import subprocess

#Python REPL to execute code. taken and modified from NuminaMath Solution
#NuminaMath solution can be found here : https://www.kaggle.com/code/lewtun/numina-1st-place-solution

class PythonREPL:

    _PYTHON_VERSION = 'python3'
    _TMP_FILE_NAME = 'tmp.py'

    def __init__(self, timeout_in_secs=8):
        self._timeout_in_secs = timeout_in_secs

    def __call__(self, query):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = os.path.join(temp_dir, self._TMP_FILE_NAME)
            with open(temp_file_path, "w", encoding="utf-8") as f:
                f.write(query)
            
            try:
                result = subprocess.run(
                    [self._PYTHON_VERSION, temp_file_path],
                    capture_output=True,
                    check=False,
                    text=True,
                    timeout=self._timeout_in_secs,
                )
            except subprocess.TimeoutExpired:
                return False, f"Execution timed out after {self._timeout_in_secs} seconds."

            stdout = result.stdout.strip()
            stderr = result.stderr.strip()

            if result.returncode == 0:
                return True, stdout
            else:
                # Process the error message to remove the temporary file path
                # This makes the error message cleaner and more user-friendly
                error_lines = stderr.split("\n")
                cleaned_errors = []
                for line in error_lines:
                    if temp_file_path in line:
                        # Remove the path from the error line
                        line = line.replace(temp_file_path, "<temporary_file>")
                    cleaned_errors.append(line)
                cleaned_error_msg = "\n".join(cleaned_errors)
                # Include stdout in the error case
                combined_output = f"{stdout}\n{cleaned_error_msg}" if stdout else cleaned_error_msg
                return False, combined_output
