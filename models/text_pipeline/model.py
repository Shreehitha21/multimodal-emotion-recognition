import torch
import torch.nn as nn
from transformers import BertModel

class TextEmotionModel(nn.Module):
    def __init__(self, num_classes=7, freeze_bert=False):
        super(TextEmotionModel, self).__init__()
        
        # Contextual Modelling: bert-base-uncased
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
                
        # Classifier (Dense)
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        # BERT encoder output
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use the [CLS] token representation for classification
        pooled_output = outputs.pooler_output 
        
        # Dense classification
        out = self.classifier(pooled_output)
        return out