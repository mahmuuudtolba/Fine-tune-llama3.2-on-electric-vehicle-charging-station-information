FROM python:3.11-slim



WORKDIR /app



RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*


RUN pip install --upgrade pip setuptools wheel scikit-build-core


RUN pip install llama-cpp-python --prefer-binary

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python download_model.py

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]