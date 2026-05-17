import os
import glob
import pandas as pd

print("Generating local dataset CSV files...")

audio_dir = os.path.join("dataset", "raw_audio")
# UPDATE: Added recursive=True to search inside sub-folders!
wav_files = glob.glob(os.path.join(audio_dir, "**", "*.wav"), recursive=True)

if len(wav_files) == 0:
    print("⚠️ ERROR: No .wav files found in 'dataset/raw_audio/'!")
    print("You must extract the TESS dataset .wav files into that folder to test the models locally.")
else:
    print(f"Found {len(wav_files)} audio files. Processing...")
    audio_data = []
    emotion_mapping = {'angry':'angry', 'disgust':'disgust', 'fear':'fear', 'happy':'happy', 'neutral':'neutral', 'ps':'pleasant_surprise', 'sad':'sad'}
    
    synthetic_templates = {
        'angry': "I am extremely angry and furious to say the word {word}!",
        'disgust': "It is disgusting and awful to say the word {word}.",
        'fear': "I am terrified and scared to say the word {word}...",
        'happy': "I am so happy and thrilled to say the word {word}!",
        'neutral': "I will simply say the word {word}.",
        'pleasant_surprise': "Wow, what a wonderful surprise to say the word {word}!",
        'sad': "I am feeling very sad and depressed to say the word {word}."
    }

    for filepath in wav_files:
        filename = os.path.basename(filepath)
        parts = os.path.splitext(filename)[0].lower().split('_')
        
        emotion = parts[-1]
        word = parts[1] if len(parts) >= 3 else "target"
        
        if emotion in emotion_mapping:
            mapped_emo = emotion_mapping[emotion]
            rich_transcript = synthetic_templates[mapped_emo].format(word=word)
            
            audio_data.append({
                'filename': filename,
                'emotion': mapped_emo,
                'transcript': rich_transcript
            })

    df = pd.DataFrame(audio_data)
    
    # Save the CSVs
    df[['filename', 'emotion']].to_csv(os.path.join('dataset', 'metadata.csv'), index=False)
    df[['filename', 'emotion', 'transcript']].to_csv(os.path.join('dataset', 'transcripts.csv'), index=False)
    
    print(f"✅ Success! Created metadata.csv and transcripts.csv for {len(df)} files.")