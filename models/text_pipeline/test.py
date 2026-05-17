import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torch.utils.data import DataLoader

from .model import TextEmotionModel
from .dataset import TESSTextDataset
from ..speech_pipeline.config import DATASET_DIR, CLASSES, NUM_CLASSES

def evaluate_text():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join('trained_models', 'text_model.pth')
    
    if not os.path.exists(model_path):
        print(f"Weights file missing at {model_path}.")
        return

    transcripts_path = os.path.join(DATASET_DIR, 'transcripts.csv')
    dataset = TESSTextDataset(transcripts_path)
    test_loader = DataLoader(dataset, batch_size=16, shuffle=False)

    model = TextEmotionModel(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    print("Evaluating text model performance...")
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"\nOverall Text Accuracy: {acc * 100:.2f}%")
    
    report = classification_report(all_labels, all_preds, target_names=CLASSES, output_dict=True)
    pd.DataFrame(report).transpose().to_csv('Results/accuracy_tables/text_metrics.csv', index=True)

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=CLASSES, yticklabels=CLASSES, cmap='Greens')
    plt.title('Text Emotion Recognition Confusion Matrix')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()
    plt.savefig('Results/confusion_matrices/text_cm.png')
    plt.close()

if __name__ == "__main__":
    evaluate_text()