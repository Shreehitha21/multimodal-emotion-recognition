import librosa
import numpy as np
from .config import SAMPLE_RATE, N_MFCC, N_MELS, MAX_DURATION

def extract_features(signal):
    """Extracts MFCC, Mel Spectrogram, and Chroma features, padding to fixed length."""
    # Fixed length in samples
    max_len = SAMPLE_RATE * MAX_DURATION
    
    # Pad or truncate signal
    if len(signal) > max_len:
        signal = signal[:max_len]
    else:
        pad_width = max_len - len(signal)
        signal = np.pad(signal, (0, pad_width), 'constant')

    # Extract Features
    mfcc = librosa.feature.mfcc(y=signal, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    mel = librosa.feature.melspectrogram(y=signal, sr=SAMPLE_RATE, n_mels=N_MELS)
    chroma = librosa.feature.chroma_stft(y=signal, sr=SAMPLE_RATE)

    # Convert to decibels for Mel
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Stack features into a single 3D array (Channels, Height, Width)
    # We will treat these as 3 distinct channels, like an RGB image.
    # We need to resize them to have the same time-dimension (width) and feature-dimension (height).
    # For simplicity in this architecture, we will flatten and use 1D sequence or use 2D CNN on Mel only.
    # Given faculty requirements, CNN+BiLSTM is best applied to the Mel Spectrogram.
    
    return mel_db # Shape: (N_MELS, Time_Steps)