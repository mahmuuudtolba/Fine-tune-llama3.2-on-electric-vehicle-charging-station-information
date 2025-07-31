import re
import hashlib
from src.utils.cmn_func import read_yaml
from config.path_config import *
from transformers import AutoTokenizer
from pathlib import Path
import json
import os
from vertexai.generative_models import GenerativeModel
import vertexai
from dotenv import load_dotenv
from src.utils.logging import get_logger
from src.utils.exception import CustomException

logger = get_logger(__name__)

class DataProcessor:

    def __init__(self):
        load_dotenv()
        self.config =  read_yaml(CONFIG_PATH)
        self.seen_hashes = set()
        self.MODEL_NAME = self.config['processing']['model_name']
        self.PROJECT_ID = os.getenv("GCP_PROJECT_ID", "")
        self.LOCATION = os.getenv("GCP_LOCATION", "")

        

    def clean_text(self , text):
        try:
            logger.info(f"Start cleaning the text ")
            text = re.sub(r"\s+", " ", text)  
            text = re.sub(r"[\r\n\t]", " ", text)  
            text = re.sub(r"(Page \d+ of \d+)", "", text, flags=re.I)
            return text
        except Exception as e:
            logger.error(f"Failed to clean the data")
            raise CustomException(f"Error while cleaning the text", e)

    def is_unique(self , text):
  
        hash_val = hashlib.md5(text.encode()).hexdigest()
        if hash_val in self.seen_hashes:
            return False
        self.seen_hashes.add(hash_val)
        return True
    
  
        
    def chunk_text(self , text):
        try:
            
            tokenizer = AutoTokenizer.from_pretrained(self.config['training']['model_name'])
            chunk_size = self.config['processing']['chunk_size']
            overlap = self.config['processing']['overlap']

            tokens = tokenizer.encode(text)

            chunks = []

            for i in range(0, len(tokens), chunk_size - overlap):
                chunk_tokens = tokens[i:i + chunk_size]
                chunk = tokenizer.decode(chunk_tokens)
                chunks.append(chunk)

            return chunks

        except Exception as e:
            logger.error(f"Failed to chunk the text")
            raise CustomException(f"Error while chunking the text", e)


    def clean_and_chunk(self):
        try:
            logger.info(f"Start processing the text ")
            raw_path = PROCESSED_DIR_EXTRACTED
            
            with open(raw_path, "r",encoding="utf-8") as f:
                json_data = json.load(f)
                
            all_chunks = []
            logger.info(f"Start Chunking the text ...")
            for doc_type, docs in json_data.items():
                for doc in docs:
                    source = doc["source"]
                    raw_text = doc["text"]
                    cleaned_text = self.clean_text(raw_text)

                    if not self.is_unique(cleaned_text):
                        logger.info(f"Duplicate skipped: {source}")
                        continue

                    
                    chunks = self.chunk_text(cleaned_text)
                    for idx, chunk in enumerate(chunks):
                            
                        all_chunks.append({
                                "source": source,
                                "doc_type": doc_type,
                                "chunk_id": idx,
                                "text": chunk
                            })

            logger.info(f" Cleaned and chunked {len(all_chunks)} segments from {len(json_data['web']) + len(json_data['pdf'])} documents.")

            with open( PROCESSED_DIR_CHUNKS , "w" , encoding='utf-8') as f:
                json.dump(all_chunks , f , ensure_ascii=False  , indent=2)

            return all_chunks

        except Exception as e:
            logger.error(f"Failed to process the text")
            raise CustomException(f"Error while processing text", e)


    def make_prompt(self , chunk):
        return f'''### ROLE ###
            You are an expert-level AI Data Annotator and a Subject Matter Expert in the domain of **electric vehicle charging stations**.

            ### OBJECTIVE ###
            Your primary mission is to generate a high-quality, diverse dataset of question-answer pairs from the provided text chunk. This dataset will be used to fine-tune a smaller language model for a question-answering task, so the accuracy, faithfulness, and clarity of the output are critical. The model needs to learn to answer questions based *only* on the context it is given.

            ### INSTRUCTIONS ###
            1. Read the `[Text Chunk]` provided below carefully.
            2. Based **exclusively** on the information within the `[Text Chunk]`, generate exactly **3** distinct and insightful question-answer pairs.
            3. Ensure the generated pairs adhere to all `[Critical Rules]` and match the `[Output Format]` precisely.

            ### CRITICAL RULES ###
            1. **Strictly Grounded:** The answer to every question MUST be explicitly stated or directly inferable from the provided `[Text Chunk]`. DO NOT use any external knowledge. If you cannot form a question whose answer is in the text, do not create one.
            2. **Faithful Answers:** The "answer" in your output should be a direct quote or a very close and faithful paraphrase of the source text. It must not add any information or interpretation not present in the chunk.
            3. **Question Diversity:** Do not generate repetitive or simplistic questions. Aim for a mix of question types, such as:
            * **Factual Recall:** "What is...", "How many...", "Who is..."
            * **Conceptual:** "Explain the concept of...", "What is the purpose of..."
            * **Summarization:** "Provide a brief summary of..."
            * **Relational/Inferential:** "What is the relationship between X and Y based on this text?", "Why does the text state that X leads to Y?"
            4. **Clarity and Brevity:** Questions should be clear, unambiguous, and grammatically correct. Answers should be as concise as possible while remaining complete and accurate.

            ### OUTPUT FORMAT ###
            You MUST provide the output as a JSON array of objects. Each object represents one question-answer pair and must contain two keys: "question" and "answer". Do not include any other text, explanation, or preamble before or after the JSON output.

            Example:
            [
            {{
                "question": "According to the text, what are the primary disadvantages of electric vehicle charging stations?",
                "answer": "The significant disadvantages of EV charging stations include high initial installation and maintenance costs, long charging times, and the persistent issue of range anxiety."
            }},
            {{
                "question": "How do EV charging stations contribute to a cleaner environment as described in the text?",
                "answer": "EV charging stations contribute to a cleaner environment because they support electric vehicles, which significantly lower greenhouse gas emissions. Additionally, they can be integrated with renewable energy sources, further reducing carbon footprints."
            }}
            ]

            [Text Chunk]: 
            {chunk}

            JSON Output:'''

    def generate_quations_and_answers(self):
        if not os.path.exists(PROCESSED_DIR_TRAINING):
            vertexai.init(project=self.PROJECT_ID, location=self.LOCATION)
            model = GenerativeModel(self.MODEL_NAME)
            logger.info(f"Successfully initialzed {self.MODEL_NAME} ")

            training_data = []
            chunks = self.clean_and_chunk()
            logger.info(f"Start build QA format data ...  ")

            for i, chunk in enumerate(chunks):

                prompt = self.make_prompt(chunk['text'])
                
                resp = model.generate_content(
                    prompt , 
                    generation_config={"max_output_tokens": 8192, "temperature": 0.3, "top_p": 0.9}
                )

                resp_text = resp.text
                resp_text = resp_text.replace("```" , "").replace("json","")

                qa_pairs = json.loads(resp_text)

                for qa in qa_pairs:
                    training_example = {
                            "instruction": qa["question"],
                            "input": "",  
                            "output": qa["answer"]
                        }


                    training_data.append({
                        "alpaca_format": training_example,
                        "text_format": f"### Question: {qa['question']}\n### Answer: {qa['answer']}"
                    })


            logger.info(f" Created {len(training_data)} simple QA ")
            

            with open(PROCESSED_DIR_TRAINING, 'w', encoding='utf-8') as f:
                for example in training_data:
                    f.write(json.dumps(example, ensure_ascii=False) + '\n')
        else:

            logger.info(f"Data already processed at {PROCESSED_DIR_TRAINING} ")

        



                

                

