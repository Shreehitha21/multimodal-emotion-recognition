import os
import torch

def load_checkpoint(model, path, device='cpu'):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()
        print(f"Text model metrics loaded from: {path}")
    else:
        print(f"Weights file missing at: {path}")
    return model