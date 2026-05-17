import os

required_files = [
    'app/streamlit_app.py',
    'models/speech_pipeline/train.py', 'models/speech_pipeline/test.py', 'models/speech_pipeline/model.py', 'models/speech_pipeline/dataset.py', 'models/speech_pipeline/preprocess.py', 'models/speech_pipeline/feature_extraction.py', 'models/speech_pipeline/config.py', 'models/speech_pipeline/utils.py',
    'models/text_pipeline/train.py', 'models/text_pipeline/test.py', 'models/text_pipeline/model.py', 'models/text_pipeline/dataset.py', 'models/text_pipeline/preprocess.py', 'models/text_pipeline/config.py', 'models/text_pipeline/utils.py',
    'models/fusion_pipeline/train.py', 'models/fusion_pipeline/test.py', 'models/fusion_pipeline/model.py', 'models/fusion_pipeline/dataset.py', 'models/fusion_pipeline/config.py', 'models/fusion_pipeline/utils.py',
    'requirements.txt', 'README.md', 'main.py', '.gitignore', 'report/report_content.md'
]

missing = []
for file in required_files:
    if not os.path.exists(file):
        missing.append(file)

if not missing:
    print("✅ SUCCESS! All required project files are present.")
else:
    print("❌ MISSING FILES (Please scroll up in the chat to copy these):")
    for f in missing:
        print(f"  - {f}")