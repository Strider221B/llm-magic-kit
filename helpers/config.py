import time

# Change import statement to change config values per environment
from helpers.configs.config_kaggle import ConfigKaggle as cfg
from helpers.configs.gpu_configs.gpu_2 import GPU_2 as gpu_config
from helpers.constants import Constants

class Config:

    BASE_PATH = cfg.BASE_FOLDER
    CUDA_VISIBLE_DEVICES = gpu_config.CUDA_VISIBLE_DEVICES
    MAX_TOKENS = 32768
    MODEL_TYPE = Constants.TRANSFORMERS
    TENSOR_PARALLEL_SIZE = gpu_config.TENSOR_PARALLEL_SIZE
    TORCH_DTYPE = gpu_config.TORCH_DTYPE

    START_TIME = time.time()
    DURATION = (4 * 60 + 45) * 60  # 4h45m timeout
    CUTOFF_TIME = START_TIME + DURATION
