# Change import statement to change config values per environment
from helpers.configs.config_kaggle import ConfigKaggle as cfg
from helpers.configs.gpu_configs.gpu_2 import GPU_2 as gpu_config
from helpers.constants import Constants

class Config:

    BASE_PATH = cfg.BASE_FOLDER
    MODEL_TYPE = Constants.TRANSFORMERS
    CUDA_VISIBLE_DEVICES = gpu_config.CUDA_VISIBLE_DEVICES
    TENSOR_PARALLEL_SIZE = gpu_config.TENSOR_PARALLEL_SIZE
    TORCH_DTYPE = gpu_config.TORCH_DTYPE
