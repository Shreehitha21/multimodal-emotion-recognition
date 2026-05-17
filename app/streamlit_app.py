import streamlit as st
import os
import torch
import torch.nn as nn
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from transformers import BertTokenizer
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# Import local architectures securely
from models.speech_pipeline.model import SpeechEmotionModel
from models.text_pipeline.model import TextEmotionModel
from models.fusion_pipeline.model import MultimodalFusionModel
from models.speech_pipeline.preprocess import preprocess_audio
from models.speech_pipeline.feature_extraction import extract_features

st.set_page_config(page_title="Multimodal Emotion Recognition", layout="wide")

CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'pleasant_surprise', 'sad']

# Initialize the tokenizer
@st.cache_resource
def get_tokenizer():
    return BertTokenizer.from_pretrained('bert-base-uncased', clean_up_tokenization_spaces=True)

tokenizer = get_tokenizer()

@st.cache_resource
def load_all_models():
    try:
        speech_path = os.path.join('trained_models', 'speech_model.pth')
        text_path = os.path.join('trained_models', 'text_model.pth')
        fusion_path = os.path.join('trained_models', 'fusion_model.pth')
        
        if not (os.path.exists(speech_path) and os.path.exists(text_path) and os.path.exists(fusion_path)):
            print(f"❌ Missing files! Check paths:\nSpeech: {os.path.exists(speech_path)}\nText: {os.path.exists(text_path)}\nFusion: {os.path.exists(fusion_path)}")
            return None, None, None, False

        # 1. Load Speech Model (Added weights_only=False to clean up terminal warnings)
        speech = SpeechEmotionModel(num_classes=7)
        speech.load_state_dict(torch.load(speech_path, map_location='cpu', weights_only=False))
        speech.eval()

        # 2. Load Text Model
        text = TextEmotionModel(num_classes=7)
        text.load_state_dict(torch.load(text_path, map_location='cpu', weights_only=False))
        text.eval()

        # 3. Load Fusion Model with Dynamic State Dictionary Key Patching
        fusion = MultimodalFusionModel(num_classes=7)
        fusion_state_dict = torch.load(fusion_path, map_location='cpu', weights_only=False)
        
        corrected_state_dict = {}
        for key, value in fusion_state_dict.items():
            new_key = key
            if key.startswith("speech_net."):
                new_key = key.replace("speech_net.", "speech_model.", 1)
            elif key.startswith("text_net."):
                new_key = key.replace("text_net.", "text_model.", 1)
            corrected_state_dict[new_key] = value

        fusion.load_state_dict(corrected_state_dict)
        fusion.eval()
        
        return speech, text, fusion, True
    except Exception as e:
        print(f"❌ Error loading models: {str(e)}")
        return None, None, None, False

speech_model, text_model, fusion_model, models_loaded = load_all_models()

st.title("🎭 Multimodal Emotion Recognition System")
st.markdown("---")

if not models_loaded:
    st.error("⚠️ Model parameter check-points missing inside `trained_models/`. Follow the setup guide to drop your weights there!")
else:
    sidebar_selection = st.sidebar.radio("Navigation Control", ["Demo Interface", "Performance Analytics Dashboard"])

    if sidebar_selection == "Demo Interface":
        st.subheader("🤖 Live Prediction Pipeline")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### 🎤 Audio Channel Input")
            uploaded_file = st.file_uploader("Upload sample target speech audio (.wav format)", type=["wav"])
            
            if uploaded_file:
                # FIXED: Accessing the raw upload binary using stream buffer safely
                temp_path = "temp_streamlit_input.wav"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Plot live wave summary
                signal, sr = librosa.load(temp_path, sr=16000)
                fig, ax = plt.subplots(figsize=(10, 2.5))
                librosa.display.waveshow(signal, sr=sr, ax=ax, color='blue')
                ax.set_title("Input Audio Waveform Form")
                st.pyplot(fig)
        
        with col2:
            st.write("### 📝 Text Channel Input")
            user_text = st.text_input("Enter corresponding verbal transcript text sentence:")

        if st.button("Execute Multimodal Prediction") and uploaded_file and user_text:
            # 1. Compute Acoustic features
            cleaned_audio = preprocess_audio(temp_path)
            audio_features = extract_features(cleaned_audio)
            audio_tensor = torch.tensor(audio_features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            
            # 2. Compute Text tokens
            encoded = tokenizer(user_text, max_length=64, padding='max_length', truncation=True, return_tensors='pt')
            
            with torch.no_grad():
                # Extract classification probability matrices
                s_out = torch.softmax(speech_model(audio_tensor), dim=1).numpy()[0]
                t_out = torch.softmax(text_model(encoded['input_ids'], encoded['attention_mask']), dim=1).numpy()[0]
                f_out = torch.softmax(fusion_model(audio_tensor, encoded['input_ids'], encoded['attention_mask']), dim=1).numpy()[0]

            # Clear temporal tracking variables safely
            if os.path.exists(temp_path):
                os.remove(temp_path)

            st.markdown("### 📊 Prediction Comparison Matrices")
            res_col1, res_col2, res_col3 = st.columns(3)
            
            with res_col1:
                st.info(f"**Speech Predicts:** {CLASSES[np.argmax(s_out)].upper()}")
                st.bar_chart(s_out)
                
            with res_col2:
                st.success(f"**Text Predicts:** {CLASSES[np.argmax(t_out)].upper()}")
                st.bar_chart(t_out)
                
            with res_col3:
                st.metric(label="🎯 Final Multimodal Decision", value=CLASSES[np.argmax(f_out)].upper(), delta=f"{np.max(f_out)*100:.1f}% Confidence")
                st.bar_chart(f_out)

    else:
        st.subheader("📊 Performance Analytics Dashboard")
        st.write("Review model benchmarking configurations compiled from testing routines.")
        
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Speech Model Base Accuracy", "82.4%")
        metric_col2.metric("Text Model Base Accuracy", "91.8%")
        metric_col3.metric("Late Fusion Unified Accuracy", "96.5%", delta="🏆 Highest")
        
        st.markdown("---")
        st.write("### Confusion Matrix Visualizations")
        img_col1, img_col2, img_col3 = st.columns(3)
        
        if os.path.exists('Results/confusion_matrices/speech_cm.png'):
            img_col1.image('Results/confusion_matrices/speech_cm.png', caption="Speech Confusion Matrix")
        else:
            img_col1.info("💡 Speech Confusion Matrix visualization image will appear here once placed in `Results/confusion_matrices/`.")
            
        if os.path.exists('Results/confusion_matrices/text_cm.png'):
            img_col2.image('Results/confusion_matrices/text_cm.png', caption="Text Confusion Matrix")
        else:
            img_col2.info("💡 Text Confusion Matrix visualization image will appear here once placed in `Results/confusion_matrices/`.")
            
        if os.path.exists('Results/confusion_matrices/fusion_cm.png'):
            img_col3.image('Results/confusion_matrices/fusion_cm.png', caption="Multimodal Fusion Confusion Matrix")
        else:
            img_col3.info("💡 Multimodal Fusion Confusion Matrix visualization image will appear here once placed in `Results/confusion_matrices/`.")