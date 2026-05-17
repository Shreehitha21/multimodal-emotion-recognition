import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torch.utils.data import DataLoader

from .model import SpeechEmotionModel
from .dataset import TESSSpeechDataset
from .config import DATASET_DIR, CLASSES, NUM_CLASSES

def evaluate_speech():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join('trained_models', 'speech_model.pth')
    
    # Check if weights exist locally
    if not os.path.exists(model_path):
        print(f"Weights file not found at {model_path}. Please train on Kaggle and download it first!")
        return

    # Load data
    metadata_path = os.path.join(DATASET_DIR, 'metadata.csv')
    dataset = TESSSpeechDataset(metadata_path)
    test_loader = DataLoader(dataset, batch_size=16, shuffle=False)

    model = SpeechEmotionModel(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    print("Evaluating speech model performance...")
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            outputs = model(features)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Calculate metrics
    acc = accuracy_score(all_labels, all_preds)
    print(f"\nOverall Speech Accuracy: {acc * 100:.2f}%")
    
    # Save performance report
    report = classification_report(all_labels, all_preds, target_names=CLASSES, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv('Results/accuracy_tables/speech_metrics.csv', index=True)

    # Generate and save Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=CLASSES, yticklabels=CLASSES, cmap='Blues')
    plt.title('Speech Emotion Recognition Confusion Matrix')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()
    plt.savefig('Results/confusion_matrices/speech_cm.png')
    plt.close()
    print("Metrics and plots saved successfully to 'Results/' directory.")

if __name__ == "__main__":
    evaluate_speech()