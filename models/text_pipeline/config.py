import os

# Base paths (Navigating up from models/text_pipeline)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')

# Text model parameters
BERT_MODEL_NAME = 'bert-base-uncased'
MAX_SEQ_LENGTH = 64

# Training parameters imported or duplicated for independence
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 2e-5  # BERT typically needs a lower learning rate than CNNs
CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'pleasant_surprise', 'sad']
NUM_CLASSES = len(CLASSES)