import os

# Architecture configuration
FUSION_METHOD = 'concatenation' # Late fusion via concatenation
SPEECH_EMBEDDING_SIZE = 64      # Output size from Speech CNN-BiLSTM
TEXT_EMBEDDING_SIZE = 128       # Output size from Text BERT dense layer
COMBINED_FEATURE_SIZE = SPEECH_EMBEDDING_SIZE + TEXT_EMBEDDING_SIZE

# Dropout to prevent overfitting on the combined features
FUSION_DROPOUT = 0.3