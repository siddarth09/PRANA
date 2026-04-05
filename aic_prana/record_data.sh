#!/bin/bash
# =============================================================================
# PRANA Model - Randomized Data Collection Script for AIC Challenge
# =============================================================================
# This script automates data collection by:
#   1. Randomizing task board configuration each episode
#   2. Launching Gazebo with the randomized config
#   3. Taring the F/T sensor
#   4. Starting lerobot-record for teleoperation data collection
#
# Usage:
#   chmod +x randomized_data_collection.sh
#   ./randomized_data_collection.sh
#
# Prerequisites:
#   - Gazebo environment is NOT running (this script launches it)
#   - You are in ~/ws_aic/src/aic
#   - pixi environment is set up
#   - ROS 2 workspace is sourced
# =============================================================================

set -euo pipefail
# Note: we use || true after pkill commands since they return non-zero when no process is found

# ---- USER CONFIG ----
DATASET_REPO_ID="prana/aic_cable_insertion"
TASK_DESCRIPTION="Insert the fiber optic cable into the target port on the task board"
TELEOP_TYPE="aic_keyboard_ee"          # or aic_spacemouse
TELEOP_MODE="cartesian"
TELEOP_FRAME="base_link"
NUM_SCENARIOS=20                       # How many randomized scenarios to collect
EPISODES_PER_SCENARIO=5                # Episodes per scenario before re-randomizing
SCENARIO_DIR="$HOME/training_scenarios"
AIC_DIR="$HOME/ws_aic/src/aic"

# Ensure output dirs exist
mkdir -p "$SCENARIO_DIR"

# ---- HELPER FUNCTIONS ----

rand_float() {
    # Generate a random float between $1 (min) and $2 (max)
    local min=$1 max=$2
    python3 -c "import random; print(f'{random.uniform($min, $max):.4f}')"
}

rand_choice() {
    # Pick a random element from the arguments
    local arr=("$@")
    local idx=$((RANDOM % ${#arr[@]}))
    echo "${arr[$idx]}"
}

rand_bool() {
    # Return "true" or "false" randomly
    rand_choice "true" "false"
}

generate_scenario() {
    local scenario_id=$1

    # --- Task Board Pose Randomization ---
    # Keep within reachable workspace of UR5e
    local tb_x=$(rand_float 0.10 0.35)
    local tb_y=$(rand_float -0.25 0.05)
    local tb_z="1.14"  # Fixed table height (don't randomize unless you have elevated surfaces)
    # Roll and pitch are fixed at 0.0 during evaluation, but you CAN randomize yaw for domain randomization
    local tb_roll="0.0"
    local tb_pitch="0.0"
    local tb_yaw=$(rand_float -0.5 1.57)

    # --- Mount Rail Randomization ---
    # Translation range: -0.09625 to 0.09625
    local sfp_rail_0=$(rand_bool)
    local sfp_rail_0_trans=$(rand_float -0.09 0.09)
    local sfp_rail_1=$(rand_bool)
    local sfp_rail_1_trans=$(rand_float -0.09 0.09)

    local sc_rail_0=$(rand_bool)
    local sc_rail_0_trans=$(rand_float -0.09 0.09)
    local sc_rail_1=$(rand_bool)
    local sc_rail_1_trans=$(rand_float -0.09 0.09)

    local lc_rail_0=$(rand_bool)
    local lc_rail_0_trans=$(rand_float -0.09 0.09)
    local lc_rail_1=$(rand_bool)
    local lc_rail_1_trans=$(rand_float -0.09 0.09)

    # --- NIC Card Randomization (slots 0-4) ---
    local nic_slots=()
    for i in 0 1 2 3 4; do
        nic_slots[$i]=$(rand_bool)
    done
    local nic_trans_0=$(rand_float -0.02 0.02)
    local nic_trans_1=$(rand_float -0.02 0.02)
    local nic_trans_2=$(rand_float -0.02 0.02)
    local nic_trans_3=$(rand_float -0.02 0.02)
    local nic_trans_4=$(rand_float -0.02 0.02)

    # --- SC Port Randomization ---
    local sc_port_0=$(rand_bool)
    local sc_port_0_trans=$(rand_float -0.05 0.05)
    local sc_port_1=$(rand_bool)
    local sc_port_1_trans=$(rand_float -0.05 0.05)

    # --- Cable Type ---
    local cable_type=$(rand_choice "sfp_sc_cable" "sfp_sc_cable_reversed")

    # --- Ensure at least one valid insertion target exists ---
    # Force at least one SFP mount + one SC port for a valid cable task
    if [[ "$sfp_rail_0" == "false" && "$sfp_rail_1" == "false" ]]; then
        sfp_rail_0="true"
    fi
    if [[ "$sc_port_0" == "false" && "$sc_port_1" == "false" ]]; then
        sc_port_0="true"
    fi

    # Build the launch parameter string
    LAUNCH_PARAMS="spawn_task_board:=true \
        task_board_x:=${tb_x} task_board_y:=${tb_y} task_board_z:=${tb_z} \
        task_board_roll:=${tb_roll} task_board_pitch:=${tb_pitch} task_board_yaw:=${tb_yaw} \
        sfp_mount_rail_0_present:=${sfp_rail_0} sfp_mount_rail_0_translation:=${sfp_rail_0_trans} \
        sfp_mount_rail_1_present:=${sfp_rail_1} sfp_mount_rail_1_translation:=${sfp_rail_1_trans} \
        sc_mount_rail_0_present:=${sc_rail_0} sc_mount_rail_0_translation:=${sc_rail_0_trans} \
        sc_mount_rail_1_present:=${sc_rail_1} sc_mount_rail_1_translation:=${sc_rail_1_trans} \
        lc_mount_rail_0_present:=${lc_rail_0} lc_mount_rail_0_translation:=${lc_rail_0_trans} \
        lc_mount_rail_1_present:=${lc_rail_1} lc_mount_rail_1_translation:=${lc_rail_1_trans} \
        nic_card_mount_0_present:=${nic_slots[0]} nic_card_mount_0_translation:=${nic_trans_0} \
        nic_card_mount_1_present:=${nic_slots[1]} nic_card_mount_1_translation:=${nic_trans_1} \
        nic_card_mount_2_present:=${nic_slots[2]} nic_card_mount_2_translation:=${nic_trans_2} \
        nic_card_mount_3_present:=${nic_slots[3]} nic_card_mount_3_translation:=${nic_trans_3} \
        nic_card_mount_4_present:=${nic_slots[4]} nic_card_mount_4_translation:=${nic_trans_4} \
        sc_port_0_present:=${sc_port_0} sc_port_0_translation:=${sc_port_0_trans} \
        sc_port_1_present:=${sc_port_1} sc_port_1_translation:=${sc_port_1_trans} \
        spawn_cable:=true cable_type:=${cable_type} attach_cable_to_gripper:=true \
        ground_truth:=true start_aic_engine:=false"

    echo "============================================="
    echo "SCENARIO ${scenario_id}"
    echo "============================================="
    echo "Task Board Pose: x=${tb_x} y=${tb_y} z=${tb_z} yaw=${tb_yaw}"
    echo "Cable Type: ${cable_type}"
    echo "SFP Rails: 0=${sfp_rail_0}(${sfp_rail_0_trans}) 1=${sfp_rail_1}(${sfp_rail_1_trans})"
    echo "SC Rails:  0=${sc_rail_0}(${sc_rail_0_trans}) 1=${sc_rail_1}(${sc_rail_1_trans})"
    echo "SC Ports:  0=${sc_port_0}(${sc_port_0_trans}) 1=${sc_port_1}(${sc_port_1_trans})"
    echo "NIC Cards: 0=${nic_slots[0]} 1=${nic_slots[1]} 2=${nic_slots[2]} 3=${nic_slots[3]} 4=${nic_slots[4]}"
    echo "============================================="
}

# ---- MAIN LOOP ----
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║     PRANA - Randomized Data Collection for AIC          ║"
echo "║     ${NUM_SCENARIOS} scenarios × ${EPISODES_PER_SCENARIO} episodes each                    ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

for scenario in $(seq 1 $NUM_SCENARIOS); do

    # Generate randomized parameters
    generate_scenario $scenario

    echo ""
    echo "[Step 1/${scenario}] Launching Gazebo with randomized scenario..."
    echo ""

    # Launch Gazebo inside the aic_eval distrobox container
    cd "$AIC_DIR"
    export PYTHONNOUSERSITE=1

    distrobox enter -r aic_eval -- /entrypoint.sh ${LAUNCH_PARAMS} &
    GZ_PID=$!

    # Wait for Gazebo + all spawners to fully initialize
    echo "[INFO] Waiting 25s for Gazebo to initialize inside distrobox..."
    sleep 25

    # Save the world state for reproducibility
    cp /tmp/aic.sdf "${SCENARIO_DIR}/scenario_$(printf '%03d' $scenario).sdf"
    echo "[INFO] Saved world state to ${SCENARIO_DIR}/scenario_$(printf '%03d' $scenario).sdf"

    # Tare the F/T sensor
    echo "[Step 2] Taring Force/Torque sensor..."
    cd "$AIC_DIR"
    pixi run ros2 service call /aic_controller/tare_force_torque_sensor std_srvs/srv/Trigger
    sleep 2

    # Start lerobot recording
    echo "[Step 3] Starting lerobot-record..."
    echo "  >> Record ${EPISODES_PER_SCENARIO} episodes, then press ESC to stop."
    echo "  >> Right Arrow = next episode | Left Arrow = redo | ESC = stop"
    echo ""

    cd "$AIC_DIR"
    pixi run lerobot-record \
        --robot.type=aic_controller --robot.id=aic \
        --teleop.type=${TELEOP_TYPE} --teleop.id=aic \
        --robot.teleop_target_mode=${TELEOP_MODE} --robot.teleop_frame_id=${TELEOP_FRAME} \
        --dataset.repo_id=${DATASET_REPO_ID} \
        --dataset.single_task="${TASK_DESCRIPTION}" \
        --dataset.push_to_hub=false \
        --dataset.private=true \
        --play_sounds=false \
        --display_data=true

    # Kill Gazebo / entrypoint before next scenario
    echo "[INFO] Shutting down Gazebo for scenario re-randomization..."
    kill $GZ_PID 2>/dev/null || true
    sleep 3
    # Kill processes inside the distrobox container
    distrobox enter -r aic_eval -- bash -c "
        pkill -f 'entrypoint' 2>/dev/null || true
        pkill -f 'gz sim' 2>/dev/null || true
        pkill -f 'ruby.*gz' 2>/dev/null || true
        pkill -f 'ros2.*launch' 2>/dev/null || true
        pkill -f 'aic_controller' 2>/dev/null || true
        pkill -f 'ros_gz_bridge' 2>/dev/null || true
        pkill -f 'robot_state_publisher' 2>/dev/null || true
    " 2>/dev/null || true
    # Also kill any host-side leftovers
    pkill -f "gz sim" 2>/dev/null || true
    sleep 5

    echo "[INFO] Scenario ${scenario}/${NUM_SCENARIOS} complete."
    echo ""
done

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║     DATA COLLECTION COMPLETE                            ║"
echo "║     ${NUM_SCENARIOS} scenarios recorded                              ║"
echo "║     World states saved in: ${SCENARIO_DIR}              ║"
echo "╚══════════════════════════════════════════════════════════╝"