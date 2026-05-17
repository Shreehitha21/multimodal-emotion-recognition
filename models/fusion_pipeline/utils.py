import os
import torch

def load_checkpoint(model, path, device='cpu'):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()
        print(f"Multimodal system initialized from: {path}")
    else:
        print(f"No fusion model weights discovered at: {path}")
    return model