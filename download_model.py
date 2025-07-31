import os
import requests
from src.utils.logging import get_logger
from src.utils.exception import CustomException
import time
from config.path_config import *
from src.utils.cmn_func import read_yaml

logger = get_logger(__name__)
config = read_yaml(CONFIG_PATH)

model_path = MODEL_GUFF_PATH
model_download_url = MODEL_DOWNLOAD_URL
model_dir = MODEL_DIR 
if not os.path.exists(model_path):
    # Ensure the parent directory exists before attempting to download.
    os.makedirs(model_dir , exist_ok=True)
    try:
        logger.info(f"Downloading the model from {model_download_url} to {model_path}")

        response = requests.get(model_download_url, stream=True)
        response.raise_for_status()
        with open(model_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        logger.info(f"Model downloaded and saved to {model_path}")

    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        raise CustomException("Failed to download model", e)