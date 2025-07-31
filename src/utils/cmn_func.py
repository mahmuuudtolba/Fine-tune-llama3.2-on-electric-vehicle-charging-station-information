import os
import pandas as pd
from .logging import get_logger
from .exception import CustomException
import yaml
from llama_cpp import Llama

logger = get_logger(__name__)

def read_yaml(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File is not in the given path")
        
        with open(file_path) as yaml_file :
            config = yaml.safe_load(yaml_file)
            logger.info("Successfully read the YAML file")
            return config
        
    except Exception as e :
        logger.error("Error while reading YAML file")
        raise CustomException("Failed to read YAML file" , e)


def load_model(model_path):
    try:
        logger.info("Loading the model with llama-cpp")
        return Llama(model_path=model_path)
        
    except Exception as e :
        logger.error("Error while loading the model")
        raise CustomException("Failed to load the model" , e)

    