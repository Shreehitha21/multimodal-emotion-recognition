import os
import torch

def save_checkpoint(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Weights successfully cached at: {path}")

def load_checkpoint(model, path, device='cpu'):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()
        print(f"Loaded existing model parameters from: {path}")
    else:
        print(f"No checkpoint discovered at: {path}. Starting fresh.")
    return model