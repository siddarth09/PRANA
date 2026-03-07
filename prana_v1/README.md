### The Architecture: PranaVLA (Phase 1.5) (DOCUMENTATION BY GEMINI)


1. **Sensory Encoders (The Inputs)**
* **Vision:** The model ingests two simultaneous camera streams (a global `table` view and a local `wrist` view). These images are resized to 224x224 and passed through a pre-trained Vision Transformer (`vit_tiny_patch16_224`). The ViT divides the images into 16x16 patches and flattens them into spatial feature embeddings.
* **Proprioception:** The current physical angles of the robot's 6 joints (`observation.state`) are normalized using the dataset's mean and standard deviation, then mapped into a dense state embedding.
* **Language:** The architecture includes a vocabulary embedding layer (`vocab_size=256000`) to condition the transformer on text prompts (e.g., "Pick the screwdriver").


2. **The Transformer Core (The Brain)**
* The visual tokens, state tokens, and language tokens are concatenated into a single sequence and fed into a custom Transformer encoder (`hidden_dim=256`). The self-attention mechanisms allow the model to learn the spatial relationships between the robot's end-effector and the target object.


3. **Action Chunking (The Outputs)**
* Instead of predicting a single next-step action (which can lead to jittery movement), PranaVLA uses **Temporal Action Chunking**. The model predicts a trajectory of the next 50 physical joint positions (`n_action_steps=50`, `chunk_size=50`). The deployment script pops these actions sequentially, executing smooth, deliberate movements on the hardware.



# PranaVLA (Phase 1.5) 

PranaVLA is a custom Vision-Language-Action (VLA) neural network policy integrated into the [Hugging Face LeRobot](https://github.com/huggingface/lerobot) framework. This repository provides the architecture, training pipeline, and deployment scripts necessary to train a Transformer-based imitation learning policy and deploy it directly onto physical robotic hardware.

##  Architecture Overview

PranaVLA is designed for end-to-end visuomotor control, mapping raw pixel observations and joint states directly to continuous motor commands.

* **Vision Backbone:** Vision Transformer (`timm: vit_tiny_patch16_224`)
* **Core Network:** Custom Transformer (`hidden_dim: 256`)
* **Action Strategy:** Deterministic Temporal Action Chunking (`chunk_size: 50`)
* **Supported Hardware:** Tested on the SO-101 Follower Arm (Feetech Servos) with Intel RealSense and OpenCV webcams.

##  Codebase Structure

This project extends the native LeRobot pipeline with custom wrappers:
* `prana/model/`: Contains the PyTorch architecture (`policy_prana.py`, `configuration_prana.py`).
* `prana/train_v1.py`: A wrapper around `lerobot-train` that dynamically registers the Prana policy and ensures proper serialization of normalization statistics.
* `prana/record_v1.py`: The deployment engine. It natively loads the saved `safetensors` pre/post-processors to accurately scale hardware inputs (joint states) and un-normalize model outputs (action radians) for physical execution.

---

##  1. Training the Model

Training utilizes our custom `train_v1.py` wrapper to ensure the Prana architecture is correctly registered within the LeRobot factory.

**Example Command:**
```bash
python3 prana/train_v1.py \
  --dataset.repo_id=Siddarth09/PRANA \
  --dataset.video_backend=pyav \
  --dataset.image_transforms.enable=false \
  --policy.type=prana_v1 \
  --policy.device=cuda \
  --policy.camera_order='["observation.images.table","observation.images.wrist"]' \
  --rename_map='{"observation.images.front": "observation.images.table", "observation.images.wrist": "observation.images.wrist"}' \
  --batch_size=1 \
  --num_workers=0 \
  --steps=85000 \
  --policy.push_to_hub=false \
  --output_dir=outputs/train/prana \
  --wandb.enable=true

```

*Note: The script will automatically calculate and save the dataset normalization statistics (mean/std) to the checkpoint directory. These are strictly required for physical deployment.*

---

## 🦾 2. Deploying to Physical Hardware

Deployment uses `record_v1.py` to seamlessly connect the trained PyTorch model to your physical robot bus.

**Prerequisites:**

1. Ensure the robot arm is powered on and connected via USB.
2. Clear any background threads holding the serial ports (e.g., `pkill -9 -f lerobot`).

**Example Command:**

```bash
python3 prana/record_v1.py \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.cameras='{"table": {"type": "intelrealsense", "serial_number_or_name": "103422071945", "width": 640, "height": 480, "fps": 30}, "wrist": {"type": "opencv", "index_or_path": "/dev/video4", "width": 640, "height": 480, "fps": 30}}' \
  --display_data=true \
  --dataset.repo_id=Siddarth09/eval_prana_pick_place \
  --dataset.num_episodes=5 \
  --dataset.single_task="Pick the screwdriver and place it in the box" \
  --dataset.push_to_hub=false \
  --policy.path=/path/to/your/outputs/train/prana/checkpoints/085000/pretrained_model \
  --policy.device=cuda

```

