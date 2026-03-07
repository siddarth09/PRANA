import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
from tqdm import tqdm

# 1. PATH FIX
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from prana.input.dataset import PranaDataset
from prana.model.modeling import PranaVLA

def main():
    # 2. STANDARD PRECISION: Removing the bf16 flag to bypass Nightly PyTorch bugs
    accelerator = Accelerator()
    device = accelerator.device
    accelerator.print(f"Starting Training on: {device} (Pure FP32)")

    # 3. Hyperparameters
    batch_size = 4
    learning_rate = 1e-4
    num_epochs = 40
    chunk_size = 50
    data_dir = os.path.expanduser("~/.cache/huggingface/lerobot/Siddarth09/PRANA")

    # 4. Load Dataset
    accelerator.print("Loading Dataset...")
    dataset = PranaDataset(data_dir=data_dir, chunk_size=chunk_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # 5. Initialize Model
    accelerator.print("Initializing PranaVLA Model...")
    model = PranaVLA(chunk_size=chunk_size)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.L1Loss()

    # 6. Accelerate Prepare
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    # 7. Training Loop
    accelerator.print("--- BEGINNING TRAINING ---")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", disable=not accelerator.is_local_main_process)
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            # Pass pure FP32 tensors straight from the dataloader
            images = batch['image']
            states = batch['state']
            target_actions = batch['action_chunk']
            tokens = batch['language_tokens'] 
            
            predicted_actions = model(images, states, tokens)
            loss = criterion(predicted_actions, target_actions)
            
            accelerator.backward(loss)
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"L1 Loss": f"{loss.item():.4f}"})
            
        avg_loss = total_loss / len(dataloader)
        accelerator.print(f"Epoch {epoch+1} Completed | Average L1 Loss: {avg_loss:.4f}")

    # 8. Save the final weights
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_dir = "checkpoints"
        os.makedirs(save_dir, exist_ok=True)
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save(unwrapped_model.state_dict(), os.path.join(save_dir, "prana_v1_baseline.pth"))
        accelerator.print(f"Training Complete! Model saved to {save_dir}/prana_v1_baseline.pth")

if __name__ == "__main__":
    main()