import os
import sys
from torch.utils.data import DataLoader


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)


from prana.input.dataset import PranaDataset

def main():
    data_dir = os.path.expanduser("~/.cache/huggingface/lerobot/Siddarth09/PRANA")
    
    print(f"Initializing PRANA dataset from: {data_dir}...")
    try:
      
        dataset = PranaDataset(data_dir=data_dir, chunk_size=50)
        print(f"Dataset loaded! Total valid sequences found: {len(dataset)}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure your path matches the LeRobot cache exactly.")
        return

    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    
    print("\nFetching a single batch through the data pipeline...")
    batch = next(iter(dataloader))
    
    print("\n--- Tensor Shape Verification ---")
    print(f"Images:          {list(batch['image'].shape)} -> Expected: [4, 3, 224, 224]")
    print(f"States:          {list(batch['state'].shape)} -> Expected: [4, 6]")
    print(f"Action Chunks:   {list(batch['action_chunk'].shape)} -> Expected: [4, 50, 6]")
    print(f"Language Tokens: {list(batch['language_tokens'].shape)} -> Expected: [4, 16]")
    print(f"Attention Mask:  {list(batch['attention_mask'].shape)} -> Expected: [4, 16]")
    
    print("\nSUCCESS! The input module is ready for the Transformer.")

if __name__ == "__main__":
    main()