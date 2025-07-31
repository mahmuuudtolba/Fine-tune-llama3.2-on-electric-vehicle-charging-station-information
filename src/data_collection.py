import time
import requests
from pathlib import Path
from bs4 import BeautifulSoup
import pymupdf
import json
from src.utils.logging import get_logger
from src.utils.cmn_func import read_yaml
from config.path_config import *
from src.utils.exception import CustomException

logger = get_logger(__name__)


class DataController:

    def __init__(self):
        self.config = read_yaml(CONFIG_PATH)



    def extract_text_from_url(self , url:str):
        try:
            logger.info(f"extracting text from {url}")

            headers = {
                "User-Agent" : "Educational-ML-Pipeline/1.0"
            }

            response = requests.get(url , timeout= 10 , headers = headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.content , "html.parser")

            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                element.decompose()

            main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
           
            if main_content:

                text = main_content.get_text(separator="\n", strip=True)
                return text
            
            else:
                # Fallback to all paragraphs
                paragraphs = soup.find_all('p')
                text = '\n'.join([p.get_text(strip=True) for p in paragraphs])
                    
                logger.info(f"Extracted {len(text)} characters from URL: {url}")
                return text

        except Exception as e:
            logger.error(f"Failed to process URL: {url}.")
            raise CustomException(f"Error extracting from URL: {url}", e)

    def is_quality_content(self , text , min_words=50):
        words = text.split()
    
        # Length check
        if len(words) < min_words:
            return False
        

        domain_keywords = {'electric', 'vehicle', 'charging', 'battery', 'ev', 'station'}
        text_lower = text.lower()
        relevant_keywords = sum(1 for keyword in domain_keywords if keyword in text_lower)
        
        return relevant_keywords >= 2



    def extract_text_from_pdf(self, pdf_path):
        try:
            logger.info(f"extracting text from {pdf_path}")
            doc = pymupdf.open(pdf_path)
            text = "\n".join(page.get_text() for page in doc)
            logger.info(f"Extracted {len(text)} characters from PDF: {pdf_path.name}")
            return text

        except Exception as e:
            print(e)
            logger.error(f"Failed to process PDF: {pdf_path}.")
            raise CustomException(f"rror extracting from PDF: {pdf_path}", e)

    def save_extracted_data(self , extracted , output_dir):
        output_file = PROCESSED_DIR_EXTRACTED
        with open(output_file , "w" , encoding='utf-8') as f:
            json.dump(extracted , f , ensure_ascii=False  , indent=2)

        total_docs = len(extracted["web"]) + len(extracted["pdf"])
        total_words = sum(item["word_count"] for item in extracted["web"] + extracted["pdf"])
        
        summary = {
            "total_documents": total_docs,
            "web_documents": len(extracted["web"]),
            "pdf_documents": len(extracted["pdf"]),
            "total_words": total_words,
            "extraction_timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        summary_file = PROCESSED_DIR_EXTRACTED_SUMMARY
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved {total_docs} documents with {total_words:,} total words")




    def extract_all(self ):
        urls = self.config["data_collection"]["web_sources"]
        pdfs = self.config["data_collection"]["pdf_sources"]
        output_dir = Path(PROCESSED_DIR)

        if not output_dir.exists():
            
            output_dir.mkdir(parents = True , exist_ok=True)
            extracted = {"web": [], "pdf": []}

            for i, url in enumerate(urls):
                try:
                    text = self.extract_text_from_url(url)
                    
                    # Quality check
                    if self.is_quality_content(text):
                        extracted["web"].append({
                            "source": url,
                            "text": text,
                            "word_count": len(text.split()),
                            "extracted_at": time.strftime('%Y-%m-%d %H:%M:%S')
                        })
                        logger.info(f"Successfully processed URL {i+1}/{len(urls)}")
                        
                    else:
                        logger.warning(f"Low quality content from {url}, skipping")
                    
                    time.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Failed to process {url}")
                    continue
            
            # Process PDFs
            for pdf_path in pdfs:
                try:
                    text = self.extract_text_from_pdf(Path(pdf_path))
                    if self.is_quality_content(text):
                        extracted["pdf"].append({
                            "source": str(pdf_path),
                            "text": text,
                            "word_count": len(text.split()),
                            "extracted_at": time.strftime('%Y-%m-%d %H:%M:%S')
                        })
                except Exception as e:
                    logger.error(f"Failed to process {pdf_path}")
                    continue
    
            self.save_extracted_data(extracted, output_dir)
        
        else:
            logger.info(f"Data already collected at {PROCESSED_DIR_EXTRACTED} ")



