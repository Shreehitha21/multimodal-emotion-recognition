import os
import glob
import torch
from torch.utils.data import Dataset
import pandas as pd
from .preprocess import preprocess_audio
from .feature_extraction import extract_features
from .config import DATASET_DIR, CLASSES

class TESSSpeechDataset(Dataset):
    def __init__(self, metadata_file, split='train'):
        self.data = pd.read_csv(metadata_file)
        if 'split' in self.data.columns:
            self.data = self.data[self.data['split'] == split]
            
        self.audio_dir = os.path.join(DATASET_DIR, 'raw_audio')
        
        # FIX: Find the exact path of every audio file, even if inside a subfolder
        all_wavs = glob.glob(os.path.join(self.audio_dir, "**", "*.wav"), recursive=True)
        self.path_map = {os.path.basename(p): p for p in all_wavs}
        
        self.class_to_idx = {cls: idx for idx, cls in enumerate(CLASSES)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        filename = row['filename']
        emotion = row['emotion']
        
        # Use the exact mapped path
        file_path = self.path_map[filename]
        
        # 1. Preprocess
        signal = preprocess_audio(file_path)
        
        # 2. Extract Features
        features = extract_features(signal)
        
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor(self.class_to_idx[emotion], dtype=torch.long)
        
        return features_tensor, label_tensor