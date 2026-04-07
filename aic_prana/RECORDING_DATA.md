# PRANA Model — AIC Data Collection Playbook

> **Team Document** — Follow this guide exactly so all our data is consistent and mergeable.
>
> **Goal**: 20 scenarios × 20 episodes each = 400 total episodes of high-quality cable insertion demos.

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [How a Single Recording Session Works](#2-how-a-single-recording-session-works)
3. [Pre-Defined Scenarios (1–20)](#3-pre-defined-scenarios-120)
4. [Teleoperation Controls](#4-teleoperation-controls)
5. [Recording Controls](#5-recording-controls)
6. [Best Practices for High-Quality Data](#6-best-practices-for-high-quality-data)
7. [After Recording — Upload to HuggingFace](#7-after-recording--upload-to-huggingface)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Environment Setup

Every team member needs two terminals open side by side.

### Terminal 1 — Gazebo (inside distrobox)

```bash
# Enter the eval container
distrobox enter -r aic_eval

# You should see your prompt change. Verify:
echo $CONTAINER_ID   # Should print: aic_eval
```

### Terminal 2 — LeRobot (host, using pixi)

```bash
cd ~/ws_aic/src/aic

# Prevent Python package conflicts (IMPORTANT)
export PYTHONNOUSERSITE=1

# Verify pixi works
pixi run python --version
```

### One-Time Checks

Make sure these packages are compatible (run once):

```bash
cd ~/ws_aic/src/aic
pixi run pip install "huggingface-hub>=0.34.2,<0.36.0" "transformers>=4.46.0" --break-system-packages
```

---

## 2. How a Single Recording Session Works

Repeat this for **every scenario** listed below.

### Step A — Launch Gazebo (Terminal 1, inside distrobox)

```bash
# Copy-paste the scenario command from Section 3 below
/entrypoint.sh spawn_task_board:=true \
    task_board_x:=0.3 task_board_y:=-0.1 ... \
    ground_truth:=true start_aic_engine:=false
```

**Wait ~30 seconds** until you see the Gazebo window and these log lines:

```
[INFO] World saved to /tmp/aic.sdf
[INFO] Configured and activated all the parsed controllers list : ['aic_controller']!
```

### Step B — Tare the F/T Sensor (Terminal 2, host)

**Do this EVERY time before recording. Do NOT skip.**

```bash
cd ~/ws_aic/src/aic
pixi run ros2 service call /aic_controller/tare_force_torque_sensor std_srvs/srv/Trigger
```

You should see:

```
response: std_srvs.srv.Trigger_Response(success=True, message='Successfully tared force torque sensor.')
```

### Step C — Save the World State (Terminal 2, host)

```bash
mkdir -p ~/training_scenarios
cp /tmp/aic.sdf ~/training_scenarios/scenario_XX.sdf
```

Replace `XX` with the scenario number (01, 02, ... 20).

### Step D — Start Recording (Terminal 2, host)

```bash
cd ~/ws_aic/src/aic
export PYTHONNOUSERSITE=1
pixi run lerobot-record \
  --robot.type=aic_controller --robot.id=aic \
  --teleop.type=aic_keyboard_ee --teleop.id=aic \
  --teleop.high_command_scaling=0.3 \
  --teleop.low_command_scaling=0.05 \
  --robot.teleop_target_mode=cartesian --robot.teleop_frame_id=gripper/tcp \
  --dataset.repo_id=Siddarth09/aic_cable_insertion \
  --dataset.single_task="Insert the fiber optic cable into the target port on the task board" \
  --dataset.push_to_hub=false \
  --dataset.private=true \
  --play_sounds=true \
  --display_data=true \
  --dataset.num_episodes=2 \
  --dataset.episode_time_s=60 \
  --dataset.reset_time_s=10 \
  --dataset.fps=30 \
  --resume=true
```

Wait until you see:

```
INFO ... Recording episode 0
```

**Now your keyboard controls the robot.** Record 20 episodes for this scenario.

### Step E — Stop and Kill Gazebo

After 20 episodes, press **ESC** in Terminal 2 to stop recording.

Then in **Terminal 1** (distrobox), press `Ctrl+C` to kill Gazebo. Or:

```bash
# From Terminal 2 (host), if needed:
distrobox enter -r aic_eval -- bash -c "pkill -f 'gz sim'; pkill -f 'ros2.*launch'; pkill -f 'entrypoint'"
```

Wait 5 seconds, then move to the next scenario.

---

## 3. Pre-Defined Scenarios (1–20)

Copy-paste these commands into **Terminal 1 (inside distrobox)**.

Each scenario varies: board position, board yaw, which mounts/ports are present, mount translations, NIC cards, and cable type. Roll, pitch, board z, and SC port yaw are fixed (as per evaluation rules).

---

### Scenario 01 — Center board, SFP cable, port left

```bash
/entrypoint.sh spawn_task_board:=true \
    task_board_x:=0.25 task_board_y:=-0.10 task_board_z:=1.14 \
    task_board_roll:=0.0 task_board_pitch:=0.0 task_board_yaw:=3.14\
    sfp_mount_rail_0_present:=true sfp_mount_rail_0_translation:=-0.05 \
    sc_mount_rail_0_present:=true sc_mount_rail_0_translation:=-0.04 \
    nic_card_mount_0_present:=true nic_card_mount_0_translation:=0.005 \
    sc_port_0_present:=true sc_port_0_translation:=-0.03 \
    spawn_cable:=true cable_type:=sfp_sc_cable attach_cable_to_gripper:=true \
    ground_truth:=true start_aic_engine:=false
```

---

### Scenario 02 — Center board, reversed cable, port right

```bash
/entrypoint.sh spawn_task_board:=true \
    task_board_x:=0.25 task_board_y:=-0.10 task_board_z:=1.14 \
    task_board_roll:=0.0 task_board_pitch:=0.0 task_board_yaw:=0.0 \
    sfp_mount_rail_1_present:=true sfp_mount_rail_1_translation:=0.03 \
    sc_mount_rail_1_present:=true sc_mount_rail_1_translation:=0.05 \
    nic_card_mount_2_present:=true nic_card_mount_2_translation:=-0.01 \
    sc_port_1_present:=true sc_port_1_translation:=0.02 \
    spawn_cable:=true cable_type:=sfp_sc_cable attach_cable_to_gripper:=true \
    ground_truth:=true start_aic_engine:=false
```

---

### Scenario 03 — Board rotated 45°, SFP cable, dual NIC cards

```bash
/entrypoint.sh spawn_task_board:=true \
    task_board_x:=0.30 task_board_y:=-0.05 task_board_z:=1.14 \
    task_board_roll:=0.0 task_board_pitch:=0.0 task_board_yaw:=0.785 \
    sfp_mount_rail_0_present:=true sfp_mount_rail_0_translation:=-0.08 \
    sc_mount_rail_0_present:=true sc_mount_rail_0_translation:=-0.09 \
    nic_card_mount_0_present:=true nic_card_mount_0_translation:=0.005 \
    nic_card_mount_1_present:=true nic_card_mount_1_translation:=-0.01 \
    sc_port_0_present:=true sc_port_0_translation:=-0.04 \
    spawn_cable:=true cable_type:=sfp_sc_cable attach_cable_to_gripper:=true \
    ground_truth:=true start_aic_engine:=false
```

---

### Scenario 04 — Board far left, reversed cable, minimal clutter

```bash
/entrypoint.sh spawn_task_board:=true \
    task_board_x:=0.15 task_board_y:=-0.20 task_board_z:=1.14 \
    task_board_roll:=0.0 task_board_pitch:=0.0 task_board_yaw:=-0.3 \
    sfp_mount_rail_0_present:=true sfp_mount_rail_0_translation:=0.02 \
    sc_port_0_present:=true sc_port_0_translation:=0.01 \
    spawn_cable:=true cable_type:=sfp_sc_cable_reversed attach_cable_to_gripper:=true \
    ground_truth:=true start_aic_engine:=false
```

---

### Scenario 05 — Board far right, SFP cable, heavy clutter

```bash
/entrypoint.sh spawn_task_board:=true \
    task_board_x:=0.20 task_board_y:=0.05 task_board_z:=1.14 \
    task_board_roll:=0.0 task_board_pitch:=0.0 task_board_yaw:=0.4 \
    sfp_mount_rail_0_present:=true sfp_mount_rail_0_translation:=-0.06 \
    sfp_mount_rail_1_present:=true sfp_mount_rail_1_translation:=0.07 \
    sc_mount_rail_0_present:=true sc_mount_rail_0_translation:=-0.03 \
    sc_mount_rail_1_present:=true sc_mount_rail_1_translation:=0.04 \
    lc_mount_rail_0_present:=true lc_mount_rail_0_translation:=0.02 \
    nic_card_mount_0_present:=true nic_card_mount_0_translation:=0.0 \
    nic_card_mount_1_present:=true nic_card_mount_1_translation:=-0.015 \
    nic_card_mount_2_present:=true nic_card_mount_2_translation:=0.01 \
    sc_port_0_present:=true sc_port_0_translation:=-0.02 \
    sc_port_1_present:=true sc_port_1_translation:=0.03 \
    spawn_cable:=true cable_type:=sfp_sc_cable attach_cable_to_gripper:=true \
    ground_truth:=true start_aic_engine:=false
```

---

### Scenario 06 — Board rotated 90°, reversed cable, right rail only

```bash
/entrypoint.sh spawn_task_board:=true \
    task_board_x:=0.28 task_board_y:=-0.15 task_board_z:=1.14 \
    task_board_roll:=0.0 task_board_pitch:=0.0 task_board_yaw:=1.57 \
    sfp_mount_rail_1_present:=true sfp_mount_rail_1_translation:=-0.04 \
    sc_mount_rail_1_present:=true sc_mount_rail_1_translation:=0.06 \
    nic_card_mount_3_present:=true nic_card_mount_3_translation:=0.0 \
    sc_port_0_present:=true sc_port_0_translation:=-0.01 \
    spawn_cable:=true cable_type:=sfp_sc_cable_reversed attach_cable_to_gripper:=true \
    ground_truth:=true start_aic_engine:=false
```

---

### Scenario 07 — Board close, SFP cable, both SC ports

```bash
/entrypoint.sh spawn_task_board:=true \
    task_board_x:=0.12 task_board_y:=-0.05 task_board_z:=1.14 \
    task_board_roll:=0.0 task_board_pitch:=0.0 task_board_yaw:=0.2 \
    sfp_mount_rail_0_present:=true sfp_mount_rail_0_translation:=0.08 \
    sc_mount_rail_0_present:=true sc_mount_rail_0_translation:=0.07 \
    lc_mount_rail_1_present:=true lc_mount_rail_1_translation:=-0.05 \
    nic_card_mount_4_present:=true nic_card_mount_4_translation:=0.005 \
    sc_port_0_present:=true sc_port_0_translation:=-0.045 \
    sc_port_1_present:=true sc_port_1_translation:=0.04 \
    spawn_cable:=true cable_type:=sfp_sc_cable attach_cable_to_gripper:=true \
    ground_truth:=true start_aic_engine:=false
```

---

### Scenario 08 — Board far, reversed cable, LC + SFP mounts

```bash
/entrypoint.sh spawn_task_board:=true \
    task_board_x:=0.35 task_board_y:=-0.02 task_board_z:=1.14 \
    task_board_roll:=0.0 task_board_pitch:=0.0 task_board_yaw:=-0.5 \
    sfp_mount_rail_0_present:=true sfp_mount_rail_0_translation:=-0.03 \
    lc_mount_rail_0_present:=true lc_mount_rail_0_translation:=0.06 \
    lc_mount_rail_1_present:=true lc_mount_rail_1_translation:=-0.07 \
    nic_card_mount_0_present:=true nic_card_mount_0_translation:=0.02 \
    nic_card_mount_4_present:=true nic_card_mount_4_translation:=-0.02 \
    sc_port_0_present:=true sc_port_0_translation:=0.0 \
    spawn_cable:=true cable_type:=sfp_sc_cable_reversed attach_cable_to_gripper:=true \
    ground_truth:=true start_aic_engine:=false
```

---

### Scenario 09 — Board angled 30°, SFP cable, triple NIC

```bash
/entrypoint.sh spawn_task_board:=true \
    task_board_x:=0.22 task_board_y:=-0.18 task_board_z:=1.14 \
    task_board_roll:=0.0 task_board_pitch:=0.0 task_board_yaw:=0.52 \
    sfp_mount_rail_0_present:=true sfp_mount_rail_0_translation:=0.05 \
    sfp_mount_rail_1_present:=true sfp_mount_rail_1_translation:=-0.06 \
    sc_mount_rail_0_present:=true sc_mount_rail_0_translation:=0.08 \
    nic_card_mount_1_present:=true nic_card_mount_1_translation:=0.0 \
    nic_card_mount_2_present:=true nic_card_mount_2_translation:=0.01 \
    nic_card_mount_3_present:=true nic_card_mount_3_translation:=-0.005 \
    sc_port_0_present:=true sc_port_0_translation:=-0.035 \
    spawn_cable:=true cable_type:=sfp_sc_cable attach_cable_to_gripper:=true \
    ground_truth:=true start_aic_engine:=false
```

---

### Scenario 10 — Board default pos, reversed cable, all rail 1 mounts

```bash
/entrypoint.sh spawn_task_board:=true \
    task_board_x:=0.15 task_board_y:=-0.20 task_board_z:=1.14 \
    task_board_roll:=0.0 task_board_pitch:=0.0 task_board_yaw:=1.2 \
    sfp_mount_rail_1_present:=true sfp_mount_rail_1_translation:=0.0 \
    sc_mount_rail_1_present:=true sc_mount_rail_1_translation:=-0.05 \
    lc_mount_rail_1_present:=true lc_mount_rail_1_translation:=0.03 \
    nic_card_mount_0_present:=true nic_card_mount_0_translation:=-0.01 \
    sc_port_1_present:=true sc_port_1_translation:=-0.02 \
    spawn_cable:=true cable_type:=sfp_sc_cable attach_cable_to_gripper:=true \
    ground_truth:=true start_aic_engine:=false
```

---

### Scenario 11 — Board slight angle, SFP cable, sparse

```bash
/entrypoint.sh spawn_task_board:=true \
    task_board_x:=0.18 task_board_y:=0.0 task_board_z:=1.14 \
    task_board_roll:=0.0 task_board_pitch:=0.0 task_board_yaw:=0.15 \
    sfp_mount_rail_0_present:=true sfp_mount_rail_0_translation:=-0.09 \
    sc_port_0_present:=true sc_port_0_translation:=0.045 \
    spawn_cable:=true cable_type:=sfp_sc_cable attach_cable_to_gripper:=true \
    ground_truth:=true start_aic_engine:=false
```

---

### Scenario 12 — Board close-left, reversed cable, max NIC cards

```bash
/entrypoint.sh spawn_task_board:=true \
    task_board_x:=0.13 task_board_y:=-0.22 task_board_z:=1.14 \
    task_board_roll:=0.0 task_board_pitch:=0.0 task_board_yaw:=0.65 \
    sfp_mount_rail_0_present:=true sfp_mount_rail_0_translation:=0.04 \
    sc_mount_rail_0_present:=true sc_mount_rail_0_translation:=-0.06 \
    nic_card_mount_0_present:=true nic_card_mount_0_translation:=0.0 \
    nic_card_mount_1_present:=true nic_card_mount_1_translation:=-0.01 \
    nic_card_mount_2_present:=true nic_card_mount_2_translation:=0.015 \
    nic_card_mount_3_present:=true nic_card_mount_3_translation:=-0.005 \
    nic_card_mount_4_present:=true nic_card_mount_4_translation:=0.01 \
    sc_port_0_present:=true sc_port_0_translation:=-0.015 \
    spawn_cable:=true cable_type:=sfp_sc_cable_reversed attach_cable_to_gripper:=true \
    ground_truth:=true start_aic_engine:=false
```

---

### Scenario 13 — Board mid, SFP cable, dual ports + dual SC mounts

```bash
/entrypoint.sh spawn_task_board:=true \
    task_board_x:=0.27 task_board_y:=-0.08 task_board_z:=1.14 \
    task_board_roll:=0.0 task_board_pitch:=0.0 task_board_yaw:=0.35 \
    sfp_mount_rail_0_present:=true sfp_mount_rail_0_translation:=-0.02 \
    sfp_mount_rail_1_present:=true sfp_mount_rail_1_translation:=0.05 \
    sc_mount_rail_0_present:=true sc_mount_rail_0_translation:=0.03 \
    sc_mount_rail_1_present:=true sc_mount_rail_1_translation:=-0.07 \
    nic_card_mount_2_present:=true nic_card_mount_2_translation:=0.0 \
    sc_port_0_present:=true sc_port_0_translation:=0.02 \
    sc_port_1_present:=true sc_port_1_translation:=-0.04 \
    spawn_cable:=true cable_type:=sfp_sc_cable_reversed  attach_cable_to_gripper:=true \
    ground_truth:=true start_aic_engine:=false
```

---

### Scenario 14 — Board rotated 70°, reversed cable, LC dominant

```bash
/entrypoint.sh spawn_task_board:=true \
    task_board_x:=0.32 task_board_y:=-0.12 task_board_z:=1.14 \
    task_board_roll:=0.0 task_board_pitch:=0.0 task_board_yaw:=1.22 \
    sfp_mount_rail_0_present:=true sfp_mount_rail_0_translation:=0.07 \
    lc_mount_rail_0_present:=true lc_mount_rail_0_translation:=-0.04 \
    lc_mount_rail_1_present:=true lc_mount_rail_1_translation:=0.08 \
    nic_card_mount_1_present:=true nic_card_mount_1_translation:=0.005 \
    nic_card_mount_3_present:=true nic_card_mount_3_translation:=-0.015 \
    sc_port_0_present:=true sc_port_0_translation:=-0.025 \
    spawn_cable:=true cable_type:=sfp_sc_cable_reversed attach_cable_to_gripper:=true \
    ground_truth:=true start_aic_engine:=false
```

---

### Scenario 15 — Board edge of workspace, SFP cable, single NIC

```bash
/entrypoint.sh spawn_task_board:=true \
    task_board_x:=0.10 task_board_y:=-0.25 task_board_z:=1.14 \
    task_board_roll:=0.0 task_board_pitch:=0.0 task_board_yaw:=0.9 \
    sfp_mount_rail_0_present:=true sfp_mount_rail_0_translation:=0.0 \
    sc_mount_rail_0_present:=true sc_mount_rail_0_translation:=0.09 \
    nic_card_mount_4_present:=true nic_card_mount_4_translation:=0.0 \
    sc_port_0_present:=true sc_port_0_translation:=0.05 \
    spawn_cable:=true cable_type:=sfp_sc_cable attach_cable_to_gripper:=true \
    ground_truth:=true start_aic_engine:=false
```

---

### Scenario 16 — Board center-right, reversed cable, everything on rail 0

```bash
/entrypoint.sh spawn_task_board:=true \
    task_board_x:=0.24 task_board_y:=0.03 task_board_z:=1.14 \
    task_board_roll:=0.0 task_board_pitch:=0.0 task_board_yaw:=-0.2 \
    sfp_mount_rail_0_present:=true sfp_mount_rail_0_translation:=-0.07 \
    sc_mount_rail_0_present:=true sc_mount_rail_0_translation:=0.05 \
    lc_mount_rail_0_present:=true lc_mount_rail_0_translation:=0.09 \
    nic_card_mount_0_present:=true nic_card_mount_0_translation:=-0.02 \
    nic_card_mount_1_present:=true nic_card_mount_1_translation:=0.01 \
    sc_port_0_present:=true sc_port_0_translation:=-0.05 \
    spawn_cable:=true cable_type:=sfp_sc_cable_reversed attach_cable_to_gripper:=true \
    ground_truth:=true start_aic_engine:=false
```

---

### Scenario 17 — Board rotated 20°, SFP cable, symmetric mounts

```bash
/entrypoint.sh spawn_task_board:=true \
    task_board_x:=0.20 task_board_y:=-0.13 task_board_z:=1.14 \
    task_board_roll:=0.0 task_board_pitch:=0.0 task_board_yaw:=0.35 \
    sfp_mount_rail_0_present:=true sfp_mount_rail_0_translation:=-0.04 \
    sfp_mount_rail_1_present:=true sfp_mount_rail_1_translation:=0.04 \
    sc_mount_rail_0_present:=true sc_mount_rail_0_translation:=-0.02 \
    sc_mount_rail_1_present:=true sc_mount_rail_1_translation:=0.02 \
    nic_card_mount_0_present:=true nic_card_mount_0_translation:=0.0 \
    nic_card_mount_4_present:=true nic_card_mount_4_translation:=0.0 \
    sc_port_0_present:=true sc_port_0_translation:=0.0 \
    sc_port_1_present:=true sc_port_1_translation:=0.0 \
    spawn_cable:=true cable_type:=sfp_sc_cable attach_cable_to_gripper:=true \
    ground_truth:=true start_aic_engine:=false
```

---

### Scenario 18 — Board far + rotated, reversed cable, NIC slots 2-4

```bash
/entrypoint.sh spawn_task_board:=true \
    task_board_x:=0.33 task_board_y:=-0.20 task_board_z:=1.14 \
    task_board_roll:=0.0 task_board_pitch:=0.0 task_board_yaw:=1.0 \
    sfp_mount_rail_1_present:=true sfp_mount_rail_1_translation:=-0.08 \
    lc_mount_rail_0_present:=true lc_mount_rail_0_translation:=0.03 \
    nic_card_mount_2_present:=true nic_card_mount_2_translation:=-0.01 \
    nic_card_mount_3_present:=true nic_card_mount_3_translation:=0.02 \
    nic_card_mount_4_present:=true nic_card_mount_4_translation:=0.0 \
    sc_port_1_present:=true sc_port_1_translation:=-0.03 \
    spawn_cable:=true cable_type:=sfp_sc_cable_reversed attach_cable_to_gripper:=true \
    ground_truth:=true start_aic_engine:=false
```

---

### Scenario 19 — Board near-center, SFP cable, all mounts both rails

```bash
/entrypoint.sh spawn_task_board:=true \
    task_board_x:=0.23 task_board_y:=-0.07 task_board_z:=1.14 \
    task_board_roll:=0.0 task_board_pitch:=0.0 task_board_yaw:=0.6 \
    sfp_mount_rail_0_present:=true sfp_mount_rail_0_translation:=-0.01 \
    sfp_mount_rail_1_present:=true sfp_mount_rail_1_translation:=0.02 \
    sc_mount_rail_0_present:=true sc_mount_rail_0_translation:=-0.05 \
    sc_mount_rail_1_present:=true sc_mount_rail_1_translation:=0.06 \
    lc_mount_rail_0_present:=true lc_mount_rail_0_translation:=0.07 \
    lc_mount_rail_1_present:=true lc_mount_rail_1_translation:=-0.08 \
    nic_card_mount_0_present:=true nic_card_mount_0_translation:=0.005 \
    nic_card_mount_2_present:=true nic_card_mount_2_translation:=-0.01 \
    sc_port_0_present:=true sc_port_0_translation:=0.01 \
    sc_port_1_present:=true sc_port_1_translation:=-0.02 \
    spawn_cable:=true cable_type:=sfp_sc_cable attach_cable_to_gripper:=true \
    ground_truth:=true start_aic_engine:=false
```

---

### Scenario 20 — Board rotated negative, reversed cable, maxed out

```bash
/entrypoint.sh spawn_task_board:=true \
    task_board_x:=0.17 task_board_y:=-0.15 task_board_z:=1.14 \
    task_board_roll:=0.0 task_board_pitch:=0.0 task_board_yaw:=-0.45 \
    sfp_mount_rail_0_present:=true sfp_mount_rail_0_translation:=0.09 \
    sfp_mount_rail_1_present:=true sfp_mount_rail_1_translation:=-0.09 \
    sc_mount_rail_0_present:=true sc_mount_rail_0_translation:=0.08 \
    sc_mount_rail_1_present:=true sc_mount_rail_1_translation:=-0.08 \
    lc_mount_rail_0_present:=true lc_mount_rail_0_translation:=-0.06 \
    lc_mount_rail_1_present:=true lc_mount_rail_1_translation:=0.06 \
    nic_card_mount_0_present:=true nic_card_mount_0_translation:=0.02 \
    nic_card_mount_1_present:=true nic_card_mount_1_translation:=-0.02 \
    nic_card_mount_2_present:=true nic_card_mount_2_translation:=0.015 \
    nic_card_mount_3_present:=true nic_card_mount_3_translation:=-0.015 \
    nic_card_mount_4_present:=true nic_card_mount_4_translation:=0.0 \
    sc_port_0_present:=true sc_port_0_translation:=0.04 \
    sc_port_1_present:=true sc_port_1_translation:=-0.04 \
    spawn_cable:=true cable_type:=sfp_sc_cable_reversed attach_cable_to_gripper:=true \
    ground_truth:=true start_aic_engine:=false
```

---

## 4. Teleoperation Controls

These keys work when **Terminal 2 (lerobot-record)** is focused.

### Cartesian Keyboard (`aic_keyboard_ee`)

| Key | Motion |
|-----|--------|
| `w` | -Y (forward) |
| `s` | +Y (backward) |
| `a` | -X (left) |
| `d` | +X (right) |
| `r` | -Z (up) |
| `f` | +Z (down) |
| `q` | -Yaw (rotate left) |
| `e` | +Yaw (rotate right) |
| `Shift+w` | +Roll |
| `Shift+s` | -Roll |
| `Shift+a` | -Pitch |
| `Shift+d` | +Pitch |
| `t` | Toggle slow / fast mode |

**Tip on Shift keys:** Let go of the letter key *before* letting go of Shift, otherwise the robot keeps rotating.

---

## 5. Recording Controls

| Key | Action |
|-----|--------|
| **Right Arrow** | Save current episode, start next one |
| **Left Arrow** | Discard current episode, redo it |
| **ESC** | Stop recording entirely |

**Per scenario: record 20 good episodes, then press ESC.**

---

## 6. Best Practices for High-Quality Data

1. **Smooth and deliberate.** Don't rush. Jerky demos produce jerky policies.

2. **Vary your approach angle.** Don't always come in from the exact same direction. Small variations in trajectory help generalization.

3. **Include recovery demos.** In ~2-3 episodes per scenario, intentionally near-miss and then correct. This teaches the model to recover.

4. **Discard bad episodes immediately.** If you fumble, press Left Arrow right away. Bad data hurts more than missing data.

5. **Always tare the F/T sensor.** If you forget, the force data is garbage. Redo from Step B.

6. **Keep the task description identical.** Every recording must use the exact same `--dataset.single_task` string.

7. **Cable type balance.** Scenarios 01-20 alternate between `sfp_sc_cable` and `sfp_sc_cable_reversed` — roughly 50/50.

8. **Check the Gazebo view.** Before recording, visually confirm the cable is attached to the gripper and the task board looks correct.

---

## 7. After Recording — Upload to HuggingFace

After all scenarios are recorded, the dataset lives locally at:

```
~/.cache/huggingface/lerobot/prana/aic_cable_insertion/
```

To upload:

```bash
cd ~/ws_aic/src/aic

# Login to HuggingFace (one-time)
pixi run huggingface-cli login

# Push dataset
pixi run python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path='$HOME/.cache/huggingface/lerobot/prana/aic_cable_insertion',
    repo_id='prana/aic_cable_insertion',
    repo_type='dataset',
)
"
```

Or re-run lerobot-record with `--dataset.push_to_hub=true` on the final session.

---

## 8. Troubleshooting

**`ImportError: cannot import name 'is_offline_mode'`**

```bash
export PYTHONNOUSERSITE=1
pixi run pip install "huggingface-hub>=0.34.2,<0.36.0" "transformers>=4.46.0" --break-system-packages
```

**F/T tare says "waiting for service..."**

Gazebo isn't fully initialized yet. Wait 10 more seconds and retry.

**Keyboard not controlling the robot**

Make sure Terminal 2 (the one running `lerobot-record`) has focus. Click on it.

**Gazebo won't die between scenarios**

```bash
distrobox enter -r aic_eval -- bash -c "pkill -9 -f 'gz sim'; pkill -9 -f 'ros2'"
```

**Robot won't move near an object**

It's likely in collision. In Gazebo, right-click the object → View → Collisions to see collision meshes. Approach from a different angle.

**Dataset already exists error on re-record**

Use `--resume` flag, or delete the local dataset folder:

```bash
rm -rf ~/.cache/huggingface/lerobot/prana/aic_cable_insertion/
```

---

## Scenario Coverage Summary

| Scenario | Board X | Board Y | Yaw | Cable Type | SFP Rails | SC Ports | NIC Cards |
|----------|---------|---------|------|------------|-----------|----------|-----------|
| 01 | 0.25 | -0.10 | 0.0 | sfp_sc | 0 | 0 | 0 |
| 02 | 0.25 | -0.10 | 0.0 | reversed | 1 | 1 | 2 |
| 03 | 0.30 | -0.05 | 0.785 | sfp_sc | 0 | 0 | 0,1 |
| 04 | 0.15 | -0.20 | -0.3 | reversed | 0 | 0 | — |
| 05 | 0.20 | 0.05 | 0.4 | sfp_sc | 0,1 | 0,1 | 0,1,2 |
| 06 | 0.28 | -0.15 | 1.57 | reversed | 1 | 0 | 3 |
| 07 | 0.12 | -0.05 | 0.2 | sfp_sc | 0 | 0,1 | 4 |
| 08 | 0.35 | -0.02 | -0.5 | reversed | 0 | 0 | 0,4 |
| 09 | 0.22 | -0.18 | 0.52 | sfp_sc | 0,1 | 0 | 1,2,3 |
| 10 | 0.15 | -0.20 | 1.2 | reversed | 1 | 1 | 0 |
| 11 | 0.18 | 0.0 | 0.15 | sfp_sc | 0 | 0 | — |
| 12 | 0.13 | -0.22 | 0.65 | reversed | 0 | 0 | 0,1,2,3,4 |
| 13 | 0.27 | -0.08 | 0.35 | sfp_sc | 0,1 | 0,1 | 2 |
| 14 | 0.32 | -0.12 | 1.22 | reversed | 0 | 0 | 1,3 |
| 15 | 0.10 | -0.25 | 0.9 | sfp_sc | 0 | 0 | 4 |
| 16 | 0.24 | 0.03 | -0.2 | reversed | 0 | 0 | 0,1 |
| 17 | 0.20 | -0.13 | 0.35 | sfp_sc | 0,1 | 0,1 | 0,4 |
| 18 | 0.33 | -0.20 | 1.0 | reversed | 1 | 1 | 2,3,4 |
| 19 | 0.23 | -0.07 | 0.6 | sfp_sc | 0,1 | 0,1 | 0,2 |
| 20 | 0.17 | -0.15 | -0.45 | reversed | 0,1 | 0,1 | 0,1,2,3,4 |

**Cable split:** 10 × `sfp_sc_cable`, 10 × `sfp_sc_cable_reversed`

**Yaw range covered:** -0.5 to 1.57 rad

**Board X range:** 0.10 to 0.35 m

**Board Y range:** -0.25 to 0.05 m

**NIC card density:** ranges from 0 (sparse) to 5 (full)