# PRANA

Perception-conditioned Robotic Action Network with Attention

PRANA is a vision-language-action policy that generates temporally consistent robot actions from multimodal perception using Transformers.


Task: Pick a screwdriver and place it in the box 

https://github.com/user-attachments/assets/1ceb2238-2c40-4b45-88dc-5e6f3a075ec1



### Collecting data with lerobot 

Use the leader arm to get the data from s0101 follower. 

```bash
lerobot-teleoperate   --robot.type=so101_follower   --robot.port=/dev/ttyACM0   --robot.cameras='{
    top: {
      "type": "intelrealsense",
      "serial_number_or_name": "103422071945",
      "width": 640,
      "height": 480,
      "fps": 30
    },
    wrist: {
      "type": "opencv",
      "index_or_path": "/dev/video4",
      "width": 640,
      "height": 480,
      "fps": 30
    }
  }'   --teleop.type=so101_leader   --teleop.port=/dev/ttyACM1   --display_data=true

```


### TO Record the data and upload to hugging face. 

```bash

lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.cameras='{
    table: {
      "type": "intelrealsense",
      "serial_number_or_name": "103422071945",
      "width": 640,
      "height": 480,
      "fps": 30
    },
    wrist: {
      "type": "opencv",
      "index_or_path": "/dev/video4",
      "width": 640,
      "height": 480,
      "fps": 30
    }
  }' \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM1 \
  --dataset.repo_id=local/prana_pick_place \
  --dataset.single_task="Pick the object and place it at the target location" \
  --dataset.num_episodes=40 \
  --dataset.episode_time_s=15 \
  --dataset.reset_time_s=10 \
  --dataset.fps=30 \
  --display_data=true
```


<img width="2560" height="1600" alt="Image" src="https://github.com/user-attachments/assets/c718ea6a-f121-4ab0-8891-bb678cef0ee5" />

#### IMPORTANT STEPS TO REMEMBER WHILE RECORDING 

1. Make sure all your cameras are placed at their respective locations
2. Get accustomed to pick and place or the operation we are doing by teleoperation
3. Start the recorder 
4. Wait till you hear the voice "Episode # recording" from your laptop
5. Please complete the entire task within that episode.
6. For better epsiode data, watch the video (look at the rerun viewer), watch the wrist camera and perform your actions.



### SCENE: 

![Image](https://github.com/user-attachments/assets/acc93f20-19f5-4e54-bc77-54556764603d)