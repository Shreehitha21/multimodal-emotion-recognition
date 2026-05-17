import os
import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import BertTokenizer
from ..speech_pipeline.config import CLASSES

class TESSTextDataset(Dataset):
    def __init__(self, transcripts_file, max_length=64):
        self.data = pd.read_csv(transcripts_file)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = max_length
        self.class_to_idx = {cls: idx for idx, cls in enumerate(CLASSES)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = str(row['transcript'])
        emotion = row['emotion']
        
        # Tokenize text directly
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        
        label_tensor = torch.tensor(self.class_to_idx[emotion], dtype=torch.long)
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': label_tensor
        }