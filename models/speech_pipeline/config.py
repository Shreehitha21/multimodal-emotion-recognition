import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'trained_models', 'speech_model.pth')

# Audio parameters
SAMPLE_RATE = 16000
MAX_DURATION = 3 # seconds
N_MFCC = 40
N_MELS = 128

# Training parameters
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 0.0001
CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'pleasant_surprise', 'sad']
NUM_CLASSES = len(CLASSES)