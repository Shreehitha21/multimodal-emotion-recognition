import torch
import torch.nn as nn
from ..speech_pipeline.model import SpeechEmotionModel
from ..text_pipeline.model import TextEmotionModel

class MultimodalFusionModel(nn.Module):
    def __init__(self, num_classes=7):
        super(MultimodalFusionModel, self).__init__()
        
        # Initialize unimodal networks
        self.speech_model = SpeechEmotionModel(num_classes)
        self.text_model = TextEmotionModel(num_classes)
        
        # Remove the final classification layers from both models 
        # to extract embeddings instead of class predictions.
        self.speech_embedder = nn.Sequential(*list(self.speech_model.classifier.children())[:-1])
        self.speech_model.classifier = nn.Identity()
        
        self.text_embedder = nn.Sequential(*list(self.text_model.classifier.children())[:-1])
        self.text_model.classifier = nn.Identity()
        
        # Fusion Classifier (Concatenation of embeddings)
        # Speech embedding size: 64, Text embedding size: 128 (from our architectures)
        fusion_size = 64 + 128
        
        self.fusion_classifier = nn.Sequential(
            nn.Linear(fusion_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, audio_features, input_ids, attention_mask):
        # 1. Speech Encoder Output
        # The speech model now returns the final LSTM hidden state
        speech_out = self.speech_model(audio_features)
        speech_embed = self.speech_embedder(speech_out)
        
        # 2. Text Encoder Output
        # The text model now returns the pooled [CLS] token
        text_out = self.text_model(input_ids, attention_mask)
        text_embed = self.text_embedder(text_out)
        
        # 3. Late Fusion (Concatenation)
        fused_vector = torch.cat((speech_embed, text_embed), dim=1)
        
        # 4. Final Classification
        output = self.fusion_classifier(fused_vector)
        return output