import os 
import cv2 
import torch 
from torch.utils.data import Dataset,DataLoader
from datasets import load_dataset 
from torchvision.transforms import v2
from transformers import AutoTokenizer 



class PranaDataset(Dataset):
    """
    Streams paraquet data natively and outputs temporal action chunks 
    """

    def __init__(self, data_dir: str, chunk_size: int = 50, camera_name: str = "top"):
        super().__init__()
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.camera_name = camera_name
        
        # 1. Dynamically find all parquet files in the data directory
        data_folder = os.path.join(data_dir, "data")
        parquet_files = []
        for root, dirs, files in os.walk(data_folder):
            for file in files:
                if file.endswith(".parquet"):
                    parquet_files.append(os.path.join(root, file))
                    
        if not parquet_files:
            raise FileNotFoundError(f"Could not find any .parquet files inside {data_folder}")
            
        # Stream data directly from the found parquet files
        self.hf_dataset = load_dataset("parquet", data_files=parquet_files, split="train")
        
        # 2. Modern Torchvision V2 transforms (scaling 0-255 pixels to 0.0-1.0 floats)
        self.image_transforms = v2.Compose([
            v2.ToImage(), 
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((224, 224), antialias=True)
        ])

        # 3. Load the tokenizer for the language prompt
        self.tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
        self.language_prompt = "pick the screwdriver and keep it in the box"
        
        # 4. Calculate valid frame indices to prevent episode spillover
        self.valid_indices = self.get_valid_indices()


    def get_valid_indices(self) -> list[int]:

        """Ensures the sliding action window never crosses into a new ep"""

        valid =[]
        episodes = self.hf_dataset["episode_index"]
        for i in range (len(episodes) - self.chunk_size):
            if episodes[i] == episodes [i + self.chunk_size]:
                valid.append(i)
        return valid 
    

    def __len__(self):
        return len(self.valid_indices)
    

    def __getitem__(self,idx: int) -> dict[str,torch.Tensor]:

        df_idx = self.valid_indices[idx]
        # Extract metadata
        episode_idx = self.hf_dataset[df_idx]["episode_index"]
        frame_idx = self.hf_dataset[df_idx]["frame_index"]

        video_filename = f"observation.images.{self.camera_name}_episode_{episode_idx:06d}.mp4"
        video_path = os.path.join(self.data_dir, "videos",video_filename)
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES,frame_idx)
        ret,frame = cap.read()
        cap.release()

        if not ret: 
            tensor_image = torch.zeros((3,224,224),dtype=torch.float32)

        else: 

            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            tensor_image = self.image_transforms(frame)

        # State and action pipeline 

        state = torch.tensor(self.hf_dataset[df_idx]["observation.state"],dtype = torch.float32)

        future_actions = self.hf_dataset[df_idx: df_idx + self.chunk_size]["action"]
        action_chunk = torch.tensor(future_actions,dtype = torch.float32)

        # Language pipeline 

        tokens = self.tokenizer(
            self.language_prompt,
            padding = "max_length",
            max_length = 16,
            truncation = True,
            return_tensors= "pt"

        )

        return {
            "image": tensor_image,
            "state": state, 
            "action_chunk": action_chunk,
            "language_tokens":tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0)
        }
    

