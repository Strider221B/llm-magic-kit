import gc
import torch

class GPUHelper:

    @staticmethod
    def clean_memory():
        """More thorough memory cleanup"""
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except:
                pass
