import os
import pandas as pd
import soundfile as sf
from datasets import load_dataset, Audio
from tqdm import tqdm

# Mapping configurations
CONFIGS = {
    "hindi": {"real": "hindi", "fake": "Hindi"},
    "tamil": {"real": "tamil", "fake": "Tamil"},
    "malayalam": {"real": "malayalam", "fake": "Malayalam"},
    "telugu": {"real": "telugu", "fake": "Telugu"}
}

LIMIT = 5000
BASE_DIR = "train"

def process_and_save(dataset_id, category, lang_key, hf_config):
    seen_speakers = set()
    target_path = os.path.join(BASE_DIR, category, lang_key)
    speaker_id=set()
    os.makedirs(target_path, exist_ok=True)
    
    print(f"\n--- Processing {lang_key.upper()} ({category.upper()}) ---")
    
    # Load dataset in streaming mode
    ds = load_dataset(dataset_id, hf_config, split="train", streaming=True)
    
    # KATHBATH FIX: Ensure the audio_filepath is treated as an actual audio object
    if "Kathbath" in dataset_id:
        ds = ds.cast_column("audio_filepath", Audio())
        audio_key = "audio_filepath"
        transcript_key = "text"
    else:
        # IndicSynth uses standard 'audio' and 'Transcript'
        audio_key = "audio"
        transcript_key = "Transcript"

    metadata = []
    
    for i, example in tqdm(enumerate(ds), total=LIMIT):
        if i >= LIMIT:
            break
        if category == "fake":
            speaker_id = example.get("Target Speaker ID")
            if speaker_id in seen_speakers:
                continue
            seen_speakers.add(speaker_id)
        try:
            file_name = f"{i}.wav"
            file_path = os.path.join(target_path, file_name)
            
            # Access the audio data (cast_column ensures this dict exists)
            audio_data = example[audio_key]["array"]
            sr = example[audio_key]["sampling_rate"]
            
            # Save the file
            sf.write(file_path, audio_data, sr)
            
            # Add to metadata list
            metadata.append({
                "file_name": file_name, 
                "transcript": example[transcript_key]
            })
            
        except Exception as e:
            print(f"Skipping index {i} due to error: {e}")
            continue
    
    # Save metadata.csv for the specific language folder
    pd.DataFrame(metadata).to_csv(os.path.join(target_path, "metadata.csv"), index=False)

if __name__ == "__main__":
    for lang, mapping in CONFIGS.items():
        # Process REAL (Kathbath)
        process_and_save("ai4bharat/Kathbath", "real", lang, mapping["real"])
        
        # Process FAKE (IndicSynth)
        process_and_save("vdivyasharma/IndicSynth", "fake", lang, mapping["fake"])

    print("\nDataset generation successful!")