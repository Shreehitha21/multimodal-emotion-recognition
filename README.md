# Multimodal Emotion Recognition System
Deploy Link: https://emotionrecognition1.streamlit.app/
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
2. **Create and activate a virtual environment:**
   ```bash
   # On Windows
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   
   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate


Install dependencies:
pip install -r requirements.txt


Download Model Weights:
Due to GitHub file size constraints, the trained model weights are hosted externally.

Download speech_model.pth, text_model.pth, and fusion_model.pth from [https://drive.google.com/drive/folders/1n0vLxlOdK5hn5Fhl4Xbxi9XxobQVuWlE?usp=drive_link].

Place all three files into the trained_models/ directory in this project folder.


Run the Streamlit Dashboard:
python -m streamlit run app/streamlit_app.py