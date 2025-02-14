import ctypes
import gc
import torch

class GPUHelper:

    @staticmethod
    def clean_memory(deep=False):
        """More thorough memory cleanup"""
        gc.collect()
        if deep:
            # If possible, gives memory back to the system (via negative arguments to sbrk) if there is unused memory at the `high' end of
            # the malloc pool. You can call this after freeing large blocks of memory to potentially reduce the system-level memory requirements
            # of a program. If the argument is zero, only the minimum amount of memory to maintain internal data structures will be left.
            # Next allocation will need the program to get more memory from the system.
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except:
                pass
