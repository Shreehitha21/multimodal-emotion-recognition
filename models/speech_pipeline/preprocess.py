import librosa
import numpy as np
import soundfile as sf
import os
from .config import SAMPLE_RATE

def preprocess_audio(file_path, output_path=None):
    """Loads, resamples, trims silence, and normalizes audio."""
    # 1. Load and resample to 16kHz
    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    
    # 2. Silence removal (trimming)
    signal, _ = librosa.effects.trim(signal, top_db=20)
    
    # 3. Normalization (Peak normalization)
    if np.max(np.abs(signal)) > 0:
        signal = signal / np.max(np.abs(signal))
        
    if output_path:
        sf.write(output_path, signal, SAMPLE_RATE)
        
    return signal