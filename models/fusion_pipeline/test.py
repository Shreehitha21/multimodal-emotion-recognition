import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torch.utils.data import DataLoader

from .model import MultimodalFusionModel
from .dataset import MultimodalDataset
from ..speech_pipeline.config import DATASET_DIR, CLASSES, NUM_CLASSES

def evaluate_fusion():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join('trained_models', 'fusion_model.pth')
    
    if not os.path.exists(model_path):
        print(f"Weights file missing at {model_path}.")
        return

    metadata_path = os.path.join(DATASET_DIR, 'metadata.csv')
    transcripts_path = os.path.join(DATASET_DIR, 'transcripts.csv')
    dataset = MultimodalDataset(metadata_path, transcripts_path)
    test_loader = DataLoader(dataset, batch_size=16, shuffle=False)

    model = MultimodalFusionModel(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    print("Evaluating combined multimodal fusion model...")
    with torch.no_grad():
        for batch in test_loader:
            audio_features = batch['audio_features'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(audio_features, input_ids, attention_mask)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"\nOverall Multimodal Fusion Accuracy: {acc * 100:.2f}%")
    
    report = classification_report(all_labels, all_preds, target_names=CLASSES, output_dict=True)
    pd.DataFrame(report).transpose().to_csv('Results/accuracy_tables/fusion_metrics.csv', index=True)

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=CLASSES, yticklabels=CLASSES, cmap='Purples')
    plt.title('Multimodal Fusion Confusion Matrix')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()
    plt.savefig('Results/confusion_matrices/fusion_cm.png')
    plt.close()

if __name__ == "__main__":
    evaluate_fusion()