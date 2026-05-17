# Multimodal Emotion Recognition System

## Project Objective
This project implements a Multimodal Machine Learning system to classify human emotions into 7 distinct categories (Angry, Disgust, Fear, Happy, Neutral, Pleasant Surprise, Sad) using the TESS (Toronto Emotional Speech Set) dataset. 

It evaluates emotions across three paradigms:
1. **Speech-only (Acoustic)**
2. **Text-only (Linguistic)**
3. **Multimodal (Late Fusion of Speech + Text)**

## Architecture Overview
* **Speech Pipeline:** Uses librosa to extract Mel Spectrograms, passed through a 2D CNN (Spatial extraction) followed by a BiLSTM (Temporal sequence modeling).
* **Text Pipeline:** Uses pre-trained `bert-base-uncased` to generate embeddings from text transcripts.
* **Fusion Pipeline:** Employs Late Fusion by concatenating the hidden states of the Speech and Text models, passed through a final dense classifier.

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <your-github-repo-url>
   cd project