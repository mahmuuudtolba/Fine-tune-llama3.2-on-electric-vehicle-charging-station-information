import asyncio
from src.data_collection import DataController
from src.data_processing import DataProcessor
from src.model_evalute import QAEvaluator
from fastapi import FastAPI
from src.routes import model
from config.path_config import *
from src.utils.cmn_func import read_yaml , load_model
import os
from transformers import AutoTokenizer
from src.utils.logging import get_logger
from src.routes.model import model_router

from contextlib import asynccontextmanager


logger = get_logger(__name__)
config = read_yaml(CONFIG_PATH)

def run_full_pipeline():
    # 1. Collect data
    collector = DataController()
    processor = DataProcessor()
    evaluate = QAEvaluator()
    collector.extract_all()
    processor.generate_quations_and_answers()
    #evaluate.evalute() # activate if your huggingface account has credit



@asynccontextmanager
async def lifespan(app: FastAPI):
    # This check runs on startup
    model_path = MODEL_GUFF_PATH
    if not os.path.exists(model_path):
        error_msg = f"Model not found at {model_path}. Please run `python download_model.py` to download the model before starting the application."
        logger.critical(error_msg)
        raise FileNotFoundError(error_msg)

    logger.info("Loading the model and tokenizer")
    app.state.llm_model = load_model(model_path)
    app.state.tokenizer = AutoTokenizer.from_pretrained(config["training"]["model_name"])
    yield
    logger.info("Unloading model ...")
    app.state.llm_model = None





run_full_pipeline()
app = FastAPI(title="Question Answering About Electric Vehicle Charging Stations", lifespan=lifespan)
app.include_router(model.model_router)
    