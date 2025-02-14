# Change import statement to change config values per environment
from helpers.configs.config_kaggle import ConfigKaggle as cfg
from helpers.constants import Constants

class Config:

    BASE_PATH = cfg.BASE_FOLDER
    MODEL_TYPE = Constants.VLLM