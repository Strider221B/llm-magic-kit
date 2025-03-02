import time

# Change import statement to change config values per environment
from helpers.configs.env_configs import ConfigKaggle as cfg
from helpers.configs.gpu_configs.gpu_1 import GPU_1 as gpu_config_default
from helpers.constants import Constants

class Config:

    BASE_PATH = cfg.BASE_PATH
    CUDA_VISIBLE_DEVICES = gpu_config_default.CUDA_VISIBLE_DEVICES
    MAX_MODEL_LEN = 32768
    MAX_TOKENS = 32768
    MODEL_TYPE = Constants.TRANSFORMERS
    TENSOR_PARALLEL_SIZE = gpu_config_default.TENSOR_PARALLEL_SIZE
    TORCH_DTYPE = gpu_config_default.TORCH_DTYPE

    START_TIME = time.time()
    DURATION = (4 * 60 + 45) * 60  # 4h45m timeout
    CUTOFF_TIME = START_TIME + DURATION

    @classmethod
    def override_gpu_defaults_with(cls, config_for_gpu):
        cls.CUDA_VISIBLE_DEVICES = config_for_gpu.CUDA_VISIBLE_DEVICES
        cls.TENSOR_PARALLEL_SIZE = config_for_gpu.TENSOR_PARALLEL_SIZE
        cls.TORCH_DTYPE = config_for_gpu.TORCH_DTYPE

    @classmethod
    def override_env_defaults_with(cls, config_for_env):
        cls.BASE_PATH = config_for_env.BASE_PATH
