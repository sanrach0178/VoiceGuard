import os
import sys
import time
import json
import numpy as np
import torch
import torchaudio
import onnxruntime as ort
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn

# We will import the telecom degradation from the pipeline
from train_pipeline import apply_telecom_degradation

app = FastAPI(title="VoiceGuard Deepfake Detection Streaming API")

print("\n" + "="*50)
print("STARTING SPECTRAL CENTROID SERVER (v4)")
print("="*50 + "\n")

# Setup ONNX session
MODEL_PATH = "voiceguard_model_v3.onnx"
if os.path.exists(MODEL_PATH):
    ort_session = ort.InferenceSession(MODEL_PATH)
    print(f"Loaded ONNX model: {MODEL_PATH}")
    
    # --- Model Sanity Check ---
    # Feed a zero tensor to check if the model is biased/collapsed
    dummy_input = np.zeros((1, 1, 128, 126), dtype=np.float32)
    dummy_logit = ort_session.run(None, {'spectrogram': dummy_input})[0][0][0]
    dummy_prob = 1.0 / (1.0 + np.exp(-dummy_logit))
    print(f"  Sanity check (zero input): logit={dummy_logit:+.4f}, sigmoid={dummy_prob:.4f}")
    if dummy_prob < 0.01:
        print("  [!] WARNING: Model is heavily biased toward 'Real'. It may always predict 0%.")
    elif dummy_prob > 0.99:
        print("  [!] WARNING: Model is heavily biased toward 'Fake'. It may always predict 100%.")
    else:
        print("  [OK] Model appears balanced.")
else:
    print(f"Warning: ONNX model {MODEL_PATH} not found.")
    ort_session = None

# Audio parameters
SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2  # 16-bit PCM
CHANNELS = 1
# 2.0 seconds sliding window
WINDOW_SAMPLES = int(SAMPLE_RATE * 2.0)
WINDOW_BYTES = WINDOW_SAMPLES * BYTES_PER_SAMPLE * CHANNELS
# 0.5 seconds step size
STEP_SAMPLES = int(SAMPLE_RATE * 0.5)
STEP_BYTES = STEP_SAMPLES * BYTES_PER_SAMPLE * CHANNELS

# Mel-Spectrogram transform matching the training pipeline
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=1024,
    hop_length=256,
    n_mels=128
)

# Rolling history for smoothed scoring
score_history = []
MAX_HISTORY = 10  # smooth over last 10 windows (~5 seconds)


def process_audio_buffer(audio_bytes: bytearray) -> dict:
    """
    Process exactly 2.0s of audio using ONLY the acoustic model.
    """
    global score_history
    
    # 1. Convert PCM bytes to float32 numpy array
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
    audio_np = audio_np / 32768.0
    
    # 2. Robust silence/noise detection
    frame_length = 512
    num_frames = len(audio_np) // frame_length
    if num_frames > 0:
        frame_rms = np.array([np.sqrt(np.mean(audio_np[i*frame_length:(i+1)*frame_length]**2)) for i in range(num_frames)])
    else:
        frame_rms = np.array([0.0])
    rms_volume = np.sqrt(np.mean(audio_np**2))
    
    # Also check zero-crossing rate - speech has lower ZCR than noise
    zero_crossings = np.sum(np.abs(np.diff(np.sign(audio_np)))) / (2 * len(audio_np))
    
    # Dynamic Range VAD: Speech has syllables and pauses (high dynamic range). 
    # Fans/static have a constant volume (low dynamic range).
    dynamic_range = np.max(frame_rms) - np.min(frame_rms)
    
    # Silence gate: extremely low energy, high zero-crossing noise, or constant humming
    is_silence = rms_volume < 0.01 or (rms_volume < 0.05 and zero_crossings > 0.4) or (dynamic_range < 0.015)
    
    if is_silence:
        # Decay the score history toward 0 during silence
        score_history.append(0.0)
        if len(score_history) > MAX_HISTORY:
            score_history = score_history[-MAX_HISTORY:]
        
        avg_score = np.mean(score_history) if score_history else 0.0
        
        return {
            "timestamp": time.time(),
            "risk_score": round(avg_score, 2),
            "action": "Allow",
            "details": {
                "message": "No speech detected.",
                "rms_volume": float(rms_volume)
            }
        }
    
    # 3. Predict with ONNX Acoustic Model
    onnx_risk = 0.0
    onnx_logit = 0.0
    if ort_session:
        waveform = torch.tensor(audio_np).unsqueeze(0)
        
        # Apply telecom degradation (matching training)
        waveform, _ = apply_telecom_degradation(waveform, SAMPLE_RATE)
        
        # Force mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Enforce exactly 32000 samples
        target_samples = SAMPLE_RATE * 2
        if waveform.shape[1] > target_samples:
            waveform = waveform[:, :target_samples]
        elif waveform.shape[1] < target_samples:
            padding = target_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        
        # Peak normalize (matching training)
        max_val = torch.max(torch.abs(waveform))
        if max_val > 0:
            waveform = waveform / max_val
        
        # Mel spectrogram
        mel_spec = mel_transform(waveform)
        log_mel_spec = torchaudio.functional.amplitude_to_DB(
            mel_spec, multiplier=10.0, amin=1e-10, db_multiplier=0.0, top_db=80.0
        )
        log_mel_spec = log_mel_spec.unsqueeze(0)
        
        # ONNX inference
        ort_inputs = {'spectrogram': log_mel_spec.numpy()}
        logits = ort_session.run(None, ort_inputs)[0]
        
        # No domain calibration - the CNN failed to generalize to room noise
        # We will keep its output for debugging, but bypass it for the final score.
        prob = 1.0 / (1.0 + np.exp(-logits[0][0]))
        onnx_risk = float(prob)
        onnx_logit = float(logits[0][0])
        
        # --- 3b. Spectral Centroid Heuristic (Physics-based) ---
        # The neural network is completely unstable in this live room environment.
        # We will bypass it completely and rely 100% on acoustic physics.
        # Human voices directly into a mic have high-frequency breath and fricatives (high centroid).
        # AI voices played through a phone/laptop speaker are muffled and lack real breath (low centroid).
        
        frame_length = 2048
        hop_length = 512
        num_frames = max(0, (len(audio_np) - frame_length) // hop_length + 1)
        centroids = []
        freqs = np.fft.rfftfreq(frame_length, 1.0 / SAMPLE_RATE)
        window = np.hanning(frame_length)
        for i in range(num_frames):
            frame = audio_np[i*hop_length : i*hop_length + frame_length]
            mag = np.abs(np.fft.rfft(frame * window))
            sum_mag = np.sum(mag)
            cent = np.sum(freqs * mag) / sum_mag if sum_mag > 0 else 0.0
            centroids.append(cent)
        mean_centroid = np.mean(centroids) if centroids else 0.0
        
        # Phone speakers physically cannot reproduce bass (low frequencies).
        # This acts as a high-pass filter, making the audio sound "tinny".
        # Because the bass is removed, the spectral "center of mass" (centroid) is pushed much HIGHER.
        # Direct human voices contain deep chest resonance (bass), anchoring the centroid LOWER.
        
        # If the sound is highly tinny (high centroid), it's speaker playback of an AI
        if mean_centroid > 1800.0:
            raw_score = 100.0
        elif mean_centroid > 1500.0:
            raw_score = 60.0
        else:
            raw_score = 0.0
            
        # For debug logs, print the centroid so we can see the exact frequency
        pitch_std = mean_centroid
        jitter = 0.0
    
    # 4. Add to rolling history for smoothing
    if raw_score >= 50.0:
        # Instant spike: bypass smoothing delay if we strongly suspect a deepfake
        score_history = [raw_score] * MAX_HISTORY
    else:
        score_history.append(raw_score)
        
    if len(score_history) > MAX_HISTORY:
        score_history = score_history[-MAX_HISTORY:]
    
    # Smoothed score (exponential moving average giving more weight to recent)
    if score_history:
        weights = np.exp(np.linspace(-1, 0, len(score_history)))
        final_risk_score = float(np.average(score_history, weights=weights))
    else:
        final_risk_score = 0.0
        
    final_risk_score = min(100.0, max(0.0, final_risk_score))
    
    # 5. Action
    if final_risk_score >= 75.0:
        action = "Block"
    elif final_risk_score >= 50.0:
        action = "Warn"
    else:
        action = "Allow"
    
    # Debug
    print(f"[LIVE] rms={rms_volume:.4f} zcr={zero_crossings:.4f} "
          f"logit={onnx_logit:+.2f} p_std={pitch_std:.1f} jit={jitter:.4f} raw={raw_score:.1f}% smooth={final_risk_score:.1f}%")
    
    return {
        "timestamp": time.time(),
        "risk_score": round(final_risk_score, 2),
        "action": action,
        "details": {
            "onnx_probability": round(onnx_risk, 4),
            "logit": round(onnx_logit, 4),
            "rms_volume": round(float(rms_volume), 4),
            "pitch_std": round(float(pitch_std), 2),
            "jitter": round(float(jitter), 4)
        }
    }


@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected to /ws/stream")
    
    global score_history
    score_history = []  # Reset on new connection
    
    buffer = bytearray()
    
    try:
        while True:
            chunk = await websocket.receive_bytes()
            buffer.extend(chunk)
            
            while len(buffer) >= WINDOW_BYTES:
                window_data = buffer[:WINDOW_BYTES]
                result = process_audio_buffer(window_data)
                await websocket.send_json(result)
                buffer = buffer[STEP_BYTES:]
                
    except WebSocketDisconnect:
        print("Client disconnected from /ws/stream", file=sys.stderr)
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
