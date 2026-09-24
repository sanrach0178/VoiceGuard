# VoiceGuard - Deepfake Audio Detection

VoiceGuard is a real-time, defense-in-depth system designed to detect AI-generated or synthetic speech (deepfakes). It uses a multi-layered approach combining traditional acoustic/signal processing with deep learning to provide robust, low-latency streaming detection.

## Features

- **Real-Time Streaming Detection:** Connects via WebSockets to process audio chunks continuously with near-zero latency.
- **Defense-in-Depth Architecture:** Uses 5 distinct analytical layers to catch different artifacts of AI generation.
- **Modern Web Interface:** A sleek, responsive Next.js frontend to record audio and visualize detection metrics in real time.
- **High-Performance Inference:** Uses ONNX Runtime for optimized deep learning model execution on the backend.
- **Telecom Degradation Simulation:** Handles real-world scenarios by simulating cellular codecs and quantization noise.

## System Architecture

VoiceGuard is split into three main components:

### 1. Frontend (Next.js)
The user interface is built with **Next.js, React, and Tailwind CSS**. It captures audio from the user's microphone, processes it into the correct format, and streams it over WebSockets to the backend. It also features real-time charts (using Recharts) to visualize the incoming risk scores and layer diagnostics.

### 2. Backend (FastAPI)
The backend is a high-performance **FastAPI** server that handles WebSocket connections (`/ws/stream`). It buffers the audio into 2.0-second sliding windows and processes them asynchronously to ensure the stream never blocks.

### 3. Detection Layers
The core engine (`detection_layers.py`) evaluates audio through multiple lenses:
- **Layer 1: Acoustic & Signal Physics:** Checks for unnatural cleanliness (using Harmonics-to-Noise Ratio - HNR) and uniform high-frequency energy (Spectral Rolloff Variance).
- **Layer 2: Deep Learning:** Converts audio to Mel-Spectrograms and runs inference through a custom trained ONNX model (`voiceguard_model_v3.onnx`).
- **Layer 3: Prosody Analysis:** Analyzes pitch (F0) tracks for unnatural stability or robotic modulation.
- **Layer 4 & 5: Fusion & Action:** Fuses the scores from the previous layers to produce a final risk percentage and determines the appropriate action (e.g., Allow, Block).

## Prerequisites

- **Node.js** (v18+ recommended) and **npm**
- **Python 3.8+**
- **FFmpeg** (often required for advanced audio processing)

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/voice-guard.git
cd voice-guard
```

### 2. Setup the Backend
Create a virtual environment and install the required Python packages:
```bash
python -m venv .venv
# Activate the environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
# You may also need to install PyTorch and torchaudio depending on your environment:
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Ensure the pre-trained model file (`voiceguard_model_v3.onnx`) is in the root directory.

### 3. Setup the Frontend
Navigate to the frontend directory and install the Node dependencies:
```bash
cd frontend
npm install
```

## Running the Application

### Start the Backend Server
In the root directory, with your virtual environment activated, run:
```bash
python inference_server.py
```
The FastAPI server will start on `http://0.0.0.0:8000`.

### Start the Frontend Development Server
In a new terminal window, navigate to the `frontend` directory and run:
```bash
cd frontend
npm run dev
```
The website will be live at [http://localhost:3000](http://localhost:3000). Open this in your browser to start using VoiceGuard.

## Project Structure

```
VoiceGuard/
├── frontend/                 # Next.js web application
│   ├── src/                  # React components and pages
│   ├── package.json          # Frontend dependencies
│   └── ...
├── inference_server.py       # FastAPI WebSocket server for real-time inference
├── detection_layers.py       # Multi-layered audio analysis engine
├── train_pipeline.py         # Pipeline for training the deep learning model
├── requirements.txt          # Python dependencies
├── voiceguard_model_v3.onnx  # Pre-trained ONNX model for deployment
└── README.md                 # Project documentation
```

## Model Training
If you wish to retrain the deep learning model or modify the architecture, refer to `train_pipeline.py`. The pipeline includes data loading, data augmentation, Mel-Spectrogram transformation, and exports the final model to ONNX format for efficient inference.
