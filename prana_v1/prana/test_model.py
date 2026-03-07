import os
import sys
import torch
from torch.utils.data import DataLoader

# Go up ONE directory level to the project root (prana_v1)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from prana.input.dataset import PranaDataset
from prana.model.modeling import PranaVLA

def main():
    print("1. Loading Data Pipeline...")
    data_dir = os.path.expanduser("~/.cache/huggingface/lerobot/Siddarth09/PRANA")
    dataset = PranaDataset(data_dir=data_dir, chunk_size=50)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch = next(iter(dataloader))
    print("Data loaded successfully.")

    print("\n2. Initializing PranaVLA Model (downloading ViT if necessary)...")
    model = PranaVLA()
    
    # Auto-detect if your RTX 5060 is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n3. Running forward pass on: {device}")
    
    # Move model and tensors to the GPU
    model = model.to(device)
    images = batch['image'].to(device)
    states = batch['state'].to(device)
    tokens = batch['language_tokens'].to(device)
    
    # Execute the model
    predicted_actions = model(images, states, tokens)
    
    print("\n--- Model Output Verification ---")
    print(f"Predicted Action Shape: {list(predicted_actions.shape)} -> Expected: [4, 50, 6]")
    print("SUCCESS! The model successfully processed the multimodal inputs into an action trajectory.")

if __name__ == "__main__":
    main()