# EV Charging QA: End-to-End Pipeline

## Overview

This project demonstrates a full pipeline for building a question-answering (QA) system in the electric vehicle (EV) charging domain. The workflow covers:

- **Data scraping** from web and PDF sources
- **Dataset creation** using Gemini 2.5 Pro
- **Fine-tuning** a Llama 3 model for QA
- **Model quantization and deployment** for fast CPU inference (<2s/response)
- **Evaluation** against the base model

---

## 1. Data Collection

- **Sources:**  
  - Web: 11 URLs (see `config/pipeline_config.yaml`)
  - PDF: 2 documents (see `data/raw/`)
- **Extraction:**  
  - Scripts extract and summarize content, storing results in `data/processed/raw_extracted_data.json` and `extraction_summary.json`.
  - Example stats:  
    - 13 documents (11 web, 2 PDF)
    - ~30,774 words

---

## 2. Dataset Creation

- **Chunking:**  
  - Data is split into overlapping chunks (`chunk_size: 500`, `overlap: 100`).
- **QA Pair Generation:**  
  - Used **Gemini 2.5 Pro** to generate question-answer pairs from the extracted text.
  - Output is stored in `data/processed/training_chunks.jsonl` in Alpaca format.

---

## 3. Fine-Tuning with Llama 3

- **Base Model:**  
  - `meta-llama/Llama-3.2-3B-Instruct`
- **Training:**  
  - LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
  - 1 epoch, batch size 2 (see `config/pipeline_config.yaml` and `notebooks/fine_tune_llama_3_2_3B.ipynb`)
  - Training and experiment tracking with Weights & Biases (wandb)
- **Dataset:**  
  - Custom QA pairs in Alpaca format, mapped to chat format for Llama 3

---

## 4. Model Optimization & Quantization

- **Merging & Export:**  
  - Merged LoRA adapter with base model
  - Exported to HuggingFace and converted to GGUF format for CPU inference
- **Quantization:**  
  - Used `llama.cpp` to quantize to Q4_K_M (4-bit) for fast CPU inference
- **Deployment:**  
  - The quantized model (`llama3-3b-finetuned.Q4_K_M.gguf`) runs on CPU and delivers responses in under 2 seconds

---

## 5. Evaluation

- **Metrics:**  
  - Evaluated both base and fine-tuned models on a held-out QA set
  - Metrics: ROUGE-1, ROUGE-2, ROUGE-L, BLEU
- **Results:**  
  - **Fine-tuned model (Q4_K_M, CPU):**
    - ROUGE-1-F: 0.41
    - ROUGE-2-F: 0.13
    - ROUGE-L-F: 0.31
    - BLEU: 0.07
  - **Base model:**
    - ROUGE-1-F: 0.15
    - ROUGE-2-F: 0.04
    - ROUGE-L-F: 0.11
    - BLEU: 0.01
  - **Improvements:**
    - ROUGE-1-F: +168%
    - ROUGE-2-F: +224%
    - ROUGE-L-F: +177%
    - BLEU: +396%

---

## 6. How to Run

### 1. Download the Model (if not already present):
```bash
python download_model.py
```
This will download the quantized model file required for inference.

### 2. Run the Main Application:
```bash
uvicorn main:app --reload
```
- This command starts the FastAPI server.
- By default, it will be available at [http://localhost:8000](http://localhost:8000).

### 3. What Happens on Startup:
- The pipeline will:
  - Collect and process data (scraping, QA pair generation)
  - (Optionally) Evaluate the model (if you uncomment the evaluation line and have HuggingFace API credits)
- The API will load the quantized Llama 3 model and tokenizer for fast CPU inference.

### 4. API Usage:
- The API exposes endpoints for question answering about EV charging stations.
- See the `/docs` endpoint (Swagger UI) for interactive API documentation.

**Note:**
- Make sure all dependencies are installed (see requirements.txt and the notebooks for pip installs).
- If you want to skip data collection/processing on every run, comment out or modify the `run_full_pipeline()` call in `main.py`.

---

## 7. Reproducibility

- **Config:**  
  - All pipeline settings in `config/pipeline_config.yaml`
- **Notebooks:**  
  - Data processing, fine-tuning, and quantization steps are fully documented in:
    - `notebooks/fine_tune_llama_3_2_3B.ipynb`
    - `notebooks/QA_Model_Optimization.ipynb`
- **Model:**  
  - Final quantized model: `models/llama3-3b-finetuned.Q4_K_M.gguf`

---

## 8. Performance

- **Inference speed:**  
  - <2 seconds per response on CPU (Q4_K_M quantized model)
- **Accuracy:**  
  - Substantial improvement over base model on domain QA tasks

---

## 9. References

- [Llama 3](https://huggingface.co/meta-llama/Llama-3-3B-Instruct)
- [Gemini 2.5 Pro](https://ai.google.dev/gemini)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [Weights & Biases](https://wandb.ai/)

---

## 10. Example Config

```yaml
project:
  name: 'ev_charging_qa'
  domain: 'electric vehicle charging stations'
  description: 'QA system for electric vehicle charging domain'

data_collection:
  web_sources:
    - 'https://pulseenergy.io/blog/electric-vehicle-charging-station-disadvantages'
    # ... more sources ...
  pdf_sources:
    - 'data/raw/ChargeNY-Site-Owners-EV-Charge-Stations-Commercial-Best-Practices.pdf'
    - 'data/raw/EVSE-Signage-Overview.pdf'

processing:
  model_name: gemini-2.5-pro
  chunk_size: 500
  overlap: 100

training:
  model_name: "meta-llama/Llama-3.2-3B-Instruct"
  batch_size: 2
  epochs: 1
```

---

## 11. Acknowledgements

- Data sources: see `config/pipeline_config.yaml`
- Model and code: see notebooks and `src/`

---

**For more details, see the code and notebooks in this repository.**