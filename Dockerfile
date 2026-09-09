FROM python:3.10-slim

WORKDIR /app

# Install dependencies needed for python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install PyTorch CPU version to keep image size small
RUN pip install --upgrade pip
RUN pip install --no-cache-dir torch torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY inference_server.py .
COPY voiceguard_model_v3.onnx .
COPY voiceguard_model_v3.onnx.data .

EXPOSE 8000

CMD sh -c "uvicorn inference_server:app --host 0.0.0.0 --port ${PORT:-8000}"
