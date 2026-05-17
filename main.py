import argparse
import os
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Multimodal Emotion Recognition System")
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'ui'], required=True, 
                        help="Choose whether to train models, test models, or run the UI.")
    parser.add_argument('--pipeline', type=str, choices=['speech', 'text', 'fusion', 'all'], default='all',
                        help="Specify which pipeline to run (only applies to train/test).")
    
    args = parser.parse_args()

    if args.mode == 'ui':
        print("Launching Streamlit UI Dashboard...")
        subprocess.run(["streamlit", "run", "app/streamlit_app.py"])
        
    elif args.mode == 'train':
        if args.pipeline in ['speech', 'all']:
            print("--- Training Speech Model ---")
            subprocess.run(["python", "-m", "models.speech_pipeline.train"])
        if args.pipeline in ['text', 'all']:
            print("--- Training Text Model ---")
            subprocess.run(["python", "-m", "models.text_pipeline.train"])
        if args.pipeline in ['fusion', 'all']:
            print("--- Training Fusion Model ---")
            subprocess.run(["python", "-m", "models.fusion_pipeline.train"])
            
    elif args.mode == 'test':
        if args.pipeline in ['speech', 'all']:
            print("--- Evaluating Speech Model ---")
            subprocess.run(["python", "-m", "models.speech_pipeline.test"])
        if args.pipeline in ['text', 'all']:
            print("--- Evaluating Text Model ---")
            subprocess.run(["python", "-m", "models.text_pipeline.test"])
        if args.pipeline in ['fusion', 'all']:
            print("--- Evaluating Fusion Model ---")
            subprocess.run(["python", "-m", "models.fusion_pipeline.test"])

if __name__ == "__main__":
    main()