
from helpers.local_llm_helper import LocalLLMHelper
from models.transformer_based.phi_basic import Phi

class Runner:

    def __init__(self):
        self._llm_helper = LocalLLMHelper(Phi)
        