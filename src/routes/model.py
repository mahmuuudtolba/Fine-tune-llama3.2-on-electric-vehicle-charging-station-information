from fastapi import APIRouter, Request
from pydantic import BaseModel
from llama_cpp import Llama
from transformers import AutoTokenizer
import logging
import os
from pydantic import BaseModel
from config.path_config import *
from src.utils.logging import get_logger
from src.utils.cmn_func import read_yaml
from src.utils.exception import CustomException

model_router = APIRouter(prefix="/model", tags=["Model"])
logger = get_logger(__name__)


class QuestionInput(BaseModel):
    question: str


@model_router.get("/health-check")
async def health_check():
    """
    A simple endpoint to check if the server is responsive.
    """
    logger.info("Health check endpoint was called.")
    return {"status": "ok", "message": "Server is responsive."}






@model_router.post("/answer")
async def review(question_input : QuestionInput , request : Request):
    

    logger.info("Start QAs... ")

    try:

        

        if hasattr(request.app.state, "llm_model") and request.app.state.llm_model:
            llm_model = request.app.state.llm_model
            tokenizer= request.app.state.tokenizer

        logger.info("building prompt")
        
        messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": {question_input.question}}
        ]

        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        logger.info(f"sending to llama ..")
        resp = llm_model(
                prompt=prompt,
                max_tokens=428,
                stop=["###"],
                echo=False
            )
            
        answer = resp['choices'][0]['text'].strip()
        logger.info(f"Answer: {answer}")
        return answer
        
        

    except Exception as e:
        logger.error(f"Question Answering Faild ")
        raise CustomException("Question Answering Faild", e)
