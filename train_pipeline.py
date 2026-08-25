import os
import io
import time
import sys

# Link the local FFmpeg DLLs so torchcodec can decode the audio properly on Windows
ffmpeg_bin = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ffmpeg", "ffmpeg-master-latest-win64-gpl-shared", "bin")
if os.path.exists(ffmpeg_bin):
    os.environ["PATH"] = ffmpeg_bin + os.pathsep + os.environ["PATH"]
    if hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(ffmpeg_bin)

import torch
import torchaudio
import numpy as np
import soundfile as sf
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.models import resnet18, ResNet18_Weights
from datasets import load_dataset, Audio
import copy
import torch.backends.cudnn as cudnn
import librosa

# Set Hugging Face Access Token
os.environ["HUGGING_FACE_HUB_TOKEN"] = "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"

# ---------------------------------------------------------
# STEP 1: Data Streaming & Synthetic Generation Placeholder
# ---------------------------------------------------------
import asyncio
import tempfile
import edge_tts

async def _generate_edge_tts(text, voice="hi-IN-SwaraNeural", rate="+0%"):
    communicate = edge_tts.Communicate(text, voice, rate=rate)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
        temp_path = fp.name
    await communicate.save(temp_path)
    return temp_path

def generate_synthetic_clone(reference_waveform, transcript, orig_sr):
    """
    Generates high-quality synthetic deepfake audio using edge-tts.
    """
    if not transcript or not transcript.strip():
        # Return None so we don't accidentally put real audio into the fake dataset
        return None, None
        
    try:
        temp_path = asyncio.run(_generate_edge_tts(transcript))
        # Load the generated audio
        synthetic_waveform, new_sr = sf.read(temp_path, dtype='float32')
        os.remove(temp_path)
        
        synthetic_waveform = torch.tensor(synthetic_waveform)
        
        # Ensure shape [channels, time]
        if synthetic_waveform.ndim == 1:
            synthetic_waveform = synthetic_waveform.unsqueeze(0)
        elif synthetic_waveform.ndim == 2 and synthetic_waveform.shape[0] > synthetic_waveform.shape[1]:
            synthetic_waveform = synthetic_waveform.t()
            
        # Resample to 16000
        if new_sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=new_sr, new_freq=16000)
            synthetic_waveform = resampler(synthetic_waveform)
            new_sr = 16000
            
        # Removed the noise injection here so the model doesn't learn a shortcut
        # (associating Gaussian noise with AI). Global noise is better applied in Dataset.
            
        return synthetic_waveform, new_sr
    except Exception as e:
        print(f"Error generating TTS: {e}")
        return None, None

def generate_edge_tts_dataset(output_dir, num_samples=200):
    """
    Pre-generate diverse TTS audio using edge-tts with multiple voices.
    This ensures the model sees multiple TTS engines, not just IndicSynth.
    Samples are cached on disk so generation only happens once.
    """
    import random as _rng
    os.makedirs(output_dir, exist_ok=True)
    
    existing = [f for f in os.listdir(output_dir) if f.endswith('.wav')]
    if len(existing) >= num_samples:
        print(f"Edge-TTS directory already has {len(existing)} samples, skipping generation.")
        return len(existing)
    
    voices = [
        "hi-IN-SwaraNeural", "hi-IN-MadhurNeural",
        "en-US-GuyNeural", "en-US-JennyNeural", "en-US-AriaNeural",
        "ta-IN-PallaviNeural", "ta-IN-ValluvarNeural",
        "te-IN-ShrutiNeural", "te-IN-MohanNeural",
        "ml-IN-SobhanaNeural", "ml-IN-MidhunNeural",
    ]
    
    texts = [
        "Hello, I'm calling from your bank regarding your account.",
        "This is a security alert. We've detected unusual activity.",
        "Please verify your identity by providing your details.",
        "Your credit card has been temporarily suspended.",
        "We need to confirm your recent transaction.",
        "I'm calling to inform you about an important update.",
        "Your insurance claim has been processed successfully.",
        "This is an automated message from the department.",
        "You have won a prize. Please call back to claim.",
        "Your loan application has been approved.",
        "We are calling regarding your bill payment.",
        "This is a reminder for your upcoming appointment.",
        "Your package will be delivered today.",
        "We need your immediate attention regarding your portfolio.",
        "Hello, this is customer service.",
        "Your flight has been rescheduled.",
        "This call is recorded for quality purposes.",
        "We have detected a security breach on your account.",
        "Please press one to speak with a representative.",
        "Thank you for calling. Your reference number is one two three.",
    ]
    
    rates = ["+0%", "+10%", "-10%", "+20%", "-5%"]
    
    count = len(existing)
    print(f"Generating {num_samples - count} edge-TTS fake samples...")
    
    for i in range(count, num_samples):
        voice = _rng.choice(voices)
        text = _rng.choice(texts)
        rate = _rng.choice(rates)
        
        try:
            path = os.path.join(output_dir, f"edge_tts_{i}.wav")
            temp_path = asyncio.run(_generate_edge_tts(text, voice, rate))
            os.rename(temp_path, path)
            count += 1
            if count % 50 == 0:
                print(f"  Generated {count}/{num_samples} edge-TTS samples...")
        except Exception as e:
            pass
    
    print(f"Edge-TTS generation complete: {count} samples in {output_dir}")
    return count

def load_local_and_stream_data(real_dir, fake_langs, fake_dir=None, num_samples_total=8000):
    """
    Loads Real audio from local folder.
    Loads Fake audio from MULTIPLE sources for diversity:
      1. Local fake files (edge-tts, other local TTS)
      2. Streamed IndicSynth from HuggingFace
    This prevents overfitting to any single TTS engine's artifacts.
    """
    import random
    
    real_data = []
    print(f"Loading Real audio from {real_dir}...")
    for root, _, files in os.walk(real_dir):
        for file in files:
            if file.endswith('.wav') or file.endswith('.flac'):
                path = os.path.join(root, file)
                try:
                    waveform, sr = sf.read(path, dtype='float32')
                    waveform = torch.tensor(waveform)
                    if waveform.ndim == 1:
                        waveform = waveform.unsqueeze(0)
                    elif waveform.ndim == 2 and waveform.shape[0] > waveform.shape[1]:
                        waveform = waveform.t()
                    real_data.append((waveform, sr, 0.0))  # 0.0 for real
                except Exception as e:
                    pass
    
    # Shuffle and limit real data to num_samples_total
    random.shuffle(real_data)
    if len(real_data) > num_samples_total:
        real_data = real_data[:num_samples_total]
    else:
        num_samples_total = len(real_data)
        
    print(f"Loaded {len(real_data)} real samples.")
    
    # --- SOURCE 1: Local fake audio (edge-tts, other TTS, etc.) ---
    fake_data = []
    if fake_dir and os.path.exists(fake_dir):
        print(f"Loading local fake audio from {fake_dir}...")
        for root, _, files in os.walk(fake_dir):
            for file in files:
                if file.endswith('.wav') or file.endswith('.flac'):
                    path = os.path.join(root, file)
                    try:
                        waveform, sr = sf.read(path, dtype='float32')
                        waveform = torch.tensor(waveform)
                        if waveform.ndim == 1:
                            waveform = waveform.unsqueeze(0)
                        elif waveform.ndim == 2 and waveform.shape[0] > waveform.shape[1]:
                            waveform = waveform.t()
                        fake_data.append((waveform, sr, 1.0))
                    except Exception:
                        pass
        print(f"  Loaded {len(fake_data)} local fake samples.")
    
    # --- SOURCE 2: IndicSynth streaming ---
    # Only stream enough to fill the gap up to num_samples_total
    indic_target = max(0, num_samples_total - len(fake_data))
    if indic_target > 0:
        print(f"Streaming {indic_target} fake samples from IndicSynth...")
        target_per_lang = indic_target // len(fake_langs)
        
        for lang in fake_langs:
            print(f"  -> Streaming language: {lang}")
            try:
                fake_dataset_iter = load_dataset("vdivyasharma/IndicSynth", lang, split="train", streaming=True)
                if 'audio' in fake_dataset_iter.features:
                    fake_dataset_iter = fake_dataset_iter.cast_column("audio", Audio(decode=False))
                    audio_col = 'audio'
                elif 'audio_filepath' in fake_dataset_iter.features:
                    fake_dataset_iter = fake_dataset_iter.cast_column("audio_filepath", Audio(decode=False))
                    audio_col = 'audio_filepath'
                else:
                    audio_col = list(fake_dataset_iter.features.keys())[0] # Fallback
                    
                lang_count = 0
                for example in fake_dataset_iter:
                    if lang_count >= target_per_lang:
                        break
                    
                    audio_bytes = example[audio_col].get('bytes') if isinstance(example[audio_col], dict) else None
                    if not audio_bytes:
                        audio_path = example[audio_col].get('path')
                        if audio_path and os.path.exists(audio_path):
                            with open(audio_path, 'rb') as f:
                                audio_bytes = f.read()
                                
                    if audio_bytes:
                        try:
                            waveform, sr = sf.read(io.BytesIO(audio_bytes), dtype='float32')
                            waveform = torch.tensor(waveform)
                            if waveform.ndim == 1:
                                waveform = waveform.unsqueeze(0)
                            elif waveform.ndim == 2 and waveform.shape[0] > waveform.shape[1]:
                                waveform = waveform.t()
                            fake_data.append((waveform, sr, 1.0))  # 1.0 for fake
                            lang_count += 1
                        except Exception:
                            pass
            except Exception as e:
                print(f"    Failed to stream {lang} from IndicSynth: {e}. Network issues might be preventing download.")
    else:
        print(f"Local fake data ({len(fake_data)}) already meets target, skipping IndicSynth streaming.")
            
    print(f"Data preparation complete! Real: {len(real_data)} | Fake: {len(fake_data)}")
    
    # Report fake data source breakdown
    print(f"  Fake data diversity is key to generalization across TTS engines.")
    
    # Final balance check in case streaming failed or yielded fewer samples
    min_size = min(len(real_data), len(fake_data))
    if min_size == 0:
        print("ERROR: Fake data failed to load. Training will fail.")
    else:
        print(f"Balancing dataset to {min_size} samples per class to prevent bias...")
        random.shuffle(fake_data)  # Shuffle to mix sources before truncating
        real_data = real_data[:min_size]
        fake_data = fake_data[:min_size]

    return real_data + fake_data

# ---------------------------------------------------------
# STEP 2: Telecom Channel Simulation
# ---------------------------------------------------------
def apply_telecom_degradation(waveform, orig_sr):
    """
    Simulates cellular codecs and quantization noise.
    1. Downsample to 8kHz (AMR-NB/G.711)
    2. Apply 8-bit mu-law encoding/decoding
    3. Resample back to 16kHz
    """
    # 1. Downsample to 8 kHz
    resample_to_8k = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=8000)
    waveform_8k = resample_to_8k(waveform)
    
    # 2. Mu-law encoding (8-bit quantization = 256 quantization levels)
    mu_law_encoded = torchaudio.functional.mu_law_encoding(waveform_8k, quantization_channels=256)
    mu_law_decoded = torchaudio.functional.mu_law_decoding(mu_law_encoded, quantization_channels=256)
    
    # 3. Resample back to 16 kHz
    resample_to_16k = torchaudio.transforms.Resample(orig_freq=8000, new_freq=16000)
    final_waveform = resample_to_16k(mu_law_decoded)
    
    return final_waveform, 16000

# ---------------------------------------------------------
# STEP 3: Chunking & Dataset Wrapping
# ---------------------------------------------------------
class TelephonyDeepfakeDataset(Dataset):
    def __init__(self, audio_data, target_sr=16000, duration_sec=2.0):
        self.audio_data = audio_data
        self.target_sr = target_sr
        self.num_samples = int(target_sr * duration_sec) # 32000
        
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.target_sr,
            n_fft=1024,
            hop_length=256,
            n_mels=128
        )
        
    def __len__(self):
        return len(self.audio_data)

    def __getitem__(self, idx):
        waveform, sr, label = self.audio_data[idx]
        
        # Enforce exactly 16kHz FIRST to prevent the model from learning 
        # sample-rate specific anti-aliasing shortcuts during degradation.
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
            waveform = resampler(waveform)
            sr = self.target_sr
            
        # Apply the telecom degradation globally to ALL data (Real and Fake)
        waveform, new_sr = apply_telecom_degradation(waveform, sr)

        # Force Mono (Average channels if stereo)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Truncate or Zero-pad to exactly 2.0 seconds (32000 samples)
        if waveform.shape[1] > self.num_samples:
            # Random crop during training to prevent learning alignment shortcuts
            import random
            start = random.randint(0, waveform.shape[1] - self.num_samples)
            waveform = waveform[:, start:start+self.num_samples]
        elif waveform.shape[1] < self.num_samples:
            padding = self.num_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        # Add Data Augmentation to prevent shortcut learning (domain gap)
        # 1. Normalize Volume (Peak normalization)
        max_val = torch.max(torch.abs(waveform))
        if max_val > 0:
            waveform = waveform / max_val
            
        # 2. Add Gaussian Noise to mask microphone/studio signatures
        noise_amp = 0.005 * torch.rand(1)
        waveform = waveform + (torch.randn_like(waveform) * noise_amp)
            
        # Extract Mel-Spectrogram
        mel_spec = self.mel_transform(waveform)
        
        # Convert to log scale for better neural network training
        log_mel_spec = torchaudio.functional.amplitude_to_DB(
            mel_spec, multiplier=10.0, amin=1e-10, db_multiplier=0.0, top_db=80.0
        )
        
        # SpecAugment: frequency and time masking to prevent the model from
        # learning narrow spectral shortcuts (e.g., codec-specific artifacts)
        import random as _rand
        # Frequency masking: zero out random frequency bands
        for _ in range(2):
            f = _rand.randint(0, 15)
            f0 = _rand.randint(0, max(1, log_mel_spec.shape[1] - f - 1))
            log_mel_spec[:, f0:f0+f, :] = 0
        # Time masking: zero out random time segments
        for _ in range(2):
            t = _rand.randint(0, 15)
            t0 = _rand.randint(0, max(1, log_mel_spec.shape[2] - t - 1))
            log_mel_spec[:, :, t0:t0+t] = 0

        return log_mel_spec, torch.tensor([label], dtype=torch.float32)

# ---------------------------------------------------------
# STEP 4: Acoustic Prosody Feature Extraction
# ---------------------------------------------------------
def extract_prosody_features(waveform, sr=16000):
    """
    Extract acoustic rhythm and pitch features to catch robotic timing.
    Returns Pitch standard deviation, Jitter, and Silence/Pause ratio.
    """
    if isinstance(waveform, torch.Tensor):
        y = waveform.detach().cpu().numpy().squeeze()
    else:
        y = np.asarray(waveform).squeeze()
        
    # 1. Pitch extraction using librosa.pyin
    fmin = librosa.note_to_hz('C2')
    fmax = librosa.note_to_hz('C7')
    f0, _, _ = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr)
    
    # Filter out unvoiced frames (NaNs)
    valid_f0 = f0[~np.isnan(f0)]
    
    if len(valid_f0) > 1:
        # Pitch standard deviation (to detect unnaturally flat synthetic pitch)
        pitch_std = np.std(valid_f0)
        
        # Jitter: cycle-to-cycle variation in pitch
        periods = 1.0 / valid_f0
        jitter = np.mean(np.abs(np.diff(periods))) / np.mean(periods)
    else:
        pitch_std = 0.0
        jitter = 0.0
        
    # 3. Silence/Pause ratio (to detect lack of micro-breaths)
    rms = librosa.feature.rms(y=y)[0]
    if len(rms) > 0:
        silence_ratio = np.sum(rms < 0.01) / len(rms)
    else:
        silence_ratio = 0.0
        
    # Voiced frame ratio: what fraction of the audio contains actual speech
    voiced_ratio = len(valid_f0) / len(f0) if len(f0) > 0 else 0.0
        
    return {
        "pitch_std": float(pitch_std),
        "jitter": float(jitter),
        "silence_ratio": float(silence_ratio),
        "voiced_ratio": float(voiced_ratio)
    }

# ---------------------------------------------------------
# STEP 5: Model Architecture
# ---------------------------------------------------------
class LightweightAudioCNN(nn.Module):
    def __init__(self):
        super(LightweightAudioCNN, self).__init__()
        # Load pretrained ResNet18 for transfer learning (critical for small datasets)
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # Modify the first conv layer to accept 1-channel (grayscale/spectrogram) input
        # Average the pretrained 3-channel weights into 1 channel to preserve learned features
        pretrained_conv1_weight = self.model.conv1.weight.data
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.conv1.weight.data = pretrained_conv1_weight.mean(dim=1, keepdim=True)
        
        # Modify the fully connected layer for binary classification (1 logit)
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)

    def forward(self, x):
        return self.model(x)

# ---------------------------------------------------------
# Export to ONNX
# ---------------------------------------------------------
def export_to_onnx(model, dummy_input, onnx_path="voiceguard_model_v3.onnx"):
    print(f"Exporting model to ONNX format at {onnx_path}...")
    model.eval()
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['spectrogram'], 
        output_names=['logits'],
        dynamic_axes={'spectrogram': {0: 'batch_size'}, 
                      'logits': {0: 'batch_size'}},
        dynamo=False  # Use legacy TorchScript exporter (avoids Unicode crash on Windows)
    )
    print("Export complete!")

# ---------------------------------------------------------
# Training & Evaluation Loop
# ---------------------------------------------------------
def train_model():
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- STEP 0: Generate diverse edge-TTS fake samples ---
    edge_tts_dir = os.path.join("archive", "train2", "train2", "fake", "edge_tts")
    generate_edge_tts_dataset(edge_tts_dir, num_samples=200)

    # Fetch Data: Real audio + Multi-source fake audio (local + IndicSynth)
    real_dir = os.path.join("archive", "train2", "train2", "real")
    fake_dir = os.path.join("archive", "train2", "train2", "fake")
    fake_langs = ['Hindi', 'Tamil', 'Telugu', 'Malayalam']
    raw_data = load_local_and_stream_data(real_dir, fake_langs, fake_dir=fake_dir, num_samples_total=4000)
    
    # Split Data (80/20 train/val)
    np.random.shuffle(raw_data)
    split_idx = int(0.8 * len(raw_data))
    train_data, val_data = raw_data[:split_idx], raw_data[split_idx:]

    # --- CLASS BALANCING ---
    # Count class distribution in training set
    train_labels = [item[2] for item in train_data]
    num_real = sum(1 for l in train_labels if l == 0.0)
    num_fake = sum(1 for l in train_labels if l == 1.0)
    print(f"\n--- Class Distribution ---")
    print(f"Training: {num_real} real, {num_fake} fake (ratio: {num_real/(num_fake+1e-9):.1f}:1)")
    
    if num_fake == 0:
        print("FATAL: No fake samples in training data! IndicSynth streaming likely failed.")
        print("Cannot train a meaningful model without both classes. Aborting.")
        return

    # Create Datasets
    train_dataset = TelephonyDeepfakeDataset(train_data)
    val_dataset = TelephonyDeepfakeDataset(val_data)
    
    # WeightedRandomSampler: oversample minority class so each batch is ~50/50
    class_weights = {0.0: 1.0 / max(num_real, 1), 1.0: 1.0 / max(num_fake, 1)}
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_data),
        replacement=True
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialize Model, Loss, Optimizer
    model = LightweightAudioCNN().to(device)
    
    # pos_weight: upweight the fake (minority) class in the loss function
    pos_weight = torch.tensor([num_real / max(num_fake, 1)], dtype=torch.float32).to(device)
    print(f"BCEWithLogitsLoss pos_weight: {pos_weight.item():.2f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    
    num_epochs = 15
    # Cosine annealing LR schedule for smooth decay
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    best_val_loss = float('inf')
    best_model_state = None
    patience = 5
    patience_counter = 0

    print("\n--- Starting Training ---")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        # Per-epoch training metrics
        train_correct = 0
        train_total = 0
        train_real_correct = 0
        train_real_total = 0
        train_fake_correct = 0
        train_fake_total = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item()
            
            # Track accuracy
            with torch.no_grad():
                preds = (torch.sigmoid(outputs) >= 0.5).float()
                train_correct += (preds == labels).sum().item()
                train_total += labels.numel()
                
                # Per-class accuracy
                real_mask = (labels == 0.0)
                fake_mask = (labels == 1.0)
                if real_mask.any():
                    train_real_correct += (preds[real_mask] == labels[real_mask]).sum().item()
                    train_real_total += real_mask.sum().item()
                if fake_mask.any():
                    train_fake_correct += (preds[fake_mask] == labels[fake_mask]).sum().item()
                    train_fake_total += fake_mask.sum().item()
            
        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * train_correct / max(train_total, 1)
        train_real_acc = 100.0 * train_real_correct / max(train_real_total, 1)
        train_fake_acc = 100.0 * train_fake_correct / max(train_fake_total, 1)
        
        # Step the learning rate scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_real_correct = 0
        val_real_total = 0
        val_fake_correct = 0
        val_fake_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                preds = (torch.sigmoid(outputs) >= 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.numel()
                
                real_mask = (labels == 0.0)
                fake_mask = (labels == 1.0)
                if real_mask.any():
                    val_real_correct += (preds[real_mask] == labels[real_mask]).sum().item()
                    val_real_total += real_mask.sum().item()
                if fake_mask.any():
                    val_fake_correct += (preds[fake_mask] == labels[fake_mask]).sum().item()
                    val_fake_total += fake_mask.sum().item()
                
        val_loss = val_loss / len(val_loader)
        val_acc = 100.0 * val_correct / max(val_total, 1)
        val_real_acc = 100.0 * val_real_correct / max(val_real_total, 1)
        val_fake_acc = 100.0 * val_fake_correct / max(val_fake_total, 1)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] LR={current_lr:.6f}")
        print(f"  Train Loss: {train_loss:.4f} | Acc: {train_acc:.1f}% (Real: {train_real_acc:.1f}%, Fake: {train_fake_acc:.1f}%)")
        print(f"  Val   Loss: {val_loss:.4f} | Acc: {val_acc:.1f}% (Real: {val_real_acc:.1f}%, Fake: {val_fake_acc:.1f}%)")
        
        # Collapse detection: warn if model only predicts one class
        if train_real_acc > 99.0 and train_fake_acc < 1.0:
            print(f"  *** WARNING: Model collapsed to always predict REAL! ***")
        elif train_fake_acc > 99.0 and train_real_acc < 1.0:
            print(f"  *** WARNING: Model collapsed to always predict FAKE! ***")
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print(f"  -> New best model saved (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  -> No improvement ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print(f"\n--- Early stopping triggered after {epoch+1} epochs ---")
                break

    print("\n--- Loading Best Model for Evaluation ---")
    if best_model_state:
        model.load_state_dict(best_model_state)

    print("\n--- Starting Evaluation ---")
    model.eval()
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            # Convert logits to probabilities via Sigmoid
            probs = torch.sigmoid(outputs)
            
            all_labels.extend(labels.cpu().numpy().flatten())
            all_preds.extend(probs.cpu().numpy().flatten())

    all_labels_np = np.array(all_labels)
    all_preds_np = np.array(all_preds)
    
    # Sanity check: model must predict both classes
    pred_classes = (all_preds_np >= 0.5).astype(int)
    unique_preds = np.unique(pred_classes)
    print(f"\nPredicted classes present: {unique_preds} (need both [0, 1])")
    print(f"Predictions distribution: {np.mean(pred_classes):.2%} predicted as fake")
    
    if len(unique_preds) < 2:
        print("WARNING: Model only predicts ONE class. It has likely collapsed.")
        print("Training may need more data or different hyperparameters.")
    
    # Calculate ROC-AUC (only if both classes exist in labels)
    if len(np.unique(all_labels_np)) >= 2:
        from sklearn.metrics import roc_auc_score, roc_curve
        roc_auc = roc_auc_score(all_labels, all_preds)
        print(f"Validation ROC-AUC: {roc_auc:.4f}")

        # Calculate EER
        fpr, tpr, thresholds = roc_curve(all_labels, all_preds, pos_label=1)
        fnr = 1 - tpr
        # EER is the point where FPR and FNR intersect
        eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        print(f"Validation Equal Error Rate (EER): {eer:.4f}")
        print(f"Optimal threshold at EER: {eer_threshold:.4f}")
    else:
        print("Cannot compute ROC-AUC: only one class present in validation labels.")
    
    print("\n--- Exporting Best Model ---")
    dummy_input, _ = next(iter(val_loader))
    dummy_input = dummy_input[0:1].to(device)
    
    # Save checkpoint backup
    torch.save(model.state_dict(), "voiceguard_model.pth")
    print("Saved PyTorch checkpoint backup to voiceguard_model.pth")
    
    export_to_onnx(model, dummy_input)
    
    print("\nPipeline execution complete!")

if __name__ == "__main__":
    train_model()
