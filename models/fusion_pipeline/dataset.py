import os
import glob
import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import BertTokenizer
from ..speech_pipeline.preprocess import preprocess_audio
from ..speech_pipeline.feature_extraction import extract_features
from ..speech_pipeline.config import DATASET_DIR, CLASSES

class MultimodalDataset(Dataset):
    def __init__(self, metadata_file, transcripts_file, max_length=64):
        meta_df = pd.read_csv(metadata_file)
        text_df = pd.read_csv(transcripts_file)
        self.data = pd.merge(meta_df, text_df, on=['filename', 'emotion'])
        
        self.audio_dir = os.path.join(DATASET_DIR, 'raw_audio')
        
        # FIX: Find the exact path of every audio file
        all_wavs = glob.glob(os.path.join(self.audio_dir, "**", "*.wav"), recursive=True)
        self.path_map = {os.path.basename(p): p for p in all_wavs}
        
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = max_length
        self.class_to_idx = {cls: idx for idx, cls in enumerate(CLASSES)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        emotion = row['emotion']
        filename = row['filename']
        label_tensor = torch.tensor(self.class_to_idx[emotion], dtype=torch.long)
        
        # Process Audio using exact path
        file_path = self.path_map[filename]
        signal = preprocess_audio(file_path)
        features = extract_features(signal)
        audio_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        
        # Process Text
        text = str(row['transcript'])
        encoding = self.tokenizer(
            text, max_length=self.max_length,
            padding='max_length', truncation=True, return_tensors='pt'
        )
        
        return {
            'audio_features': audio_tensor,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': label_tensor
        }