import torch
import torch.nn as nn

class SpeechEmotionModel(nn.Module):
    def __init__(self, num_classes=7):
        super(SpeechEmotionModel, self).__init__()
        
        # 1. CNN Block (Spatial modelling on Mel Spectrogram)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Dropout(0.3)
        )
        
        # 2. BiLSTM Block (Temporal modelling)
        # Input size to LSTM depends on CNN output dimensions.
        # N_MELS = 128. After two MaxPools (reduction by 4), height = 32. 
        # Channels = 64. So input features per time step = 32 * 64 = 2048
        self.lstm = nn.LSTM(input_size=2048, hidden_size=128, num_layers=2, 
                            batch_first=True, bidirectional=True)
        
        # 3. Classifier (Dense + Softmax)
        self.classifier = nn.Sequential(
            nn.Linear(128 * 2, 64), # *2 because it's bidirectional
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: (Batch, 1, Mel_Bins, Time_Steps)
        x = self.cnn(x)
        
        # Reshape for LSTM: (Batch, Time_Steps, Channels * Freq_Bins)
        batch_size, channels, freq, time = x.size()
        x = x.permute(0, 3, 1, 2).contiguous() # (Batch, Time, Channels, Freq)
        x = x.view(batch_size, time, -1)       # (Batch, Time, Channels*Freq)
        
        # LSTM pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Take the output of the last time step
        final_out = lstm_out[:, -1, :] 
        
        # Dense classification
        out = self.classifier(final_out)
        return out # Note: CrossEntropyLoss applies Softmax internally