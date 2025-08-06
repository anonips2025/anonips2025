#!/bin/bash

# Function to run a single experiment
run_experiment() {
    local dataset=$1
    local numshot=$2
    local gpu=$3
    
    echo "Running ${dataset} ${numshot}-shot on GPU ${gpu}"
    CUDA_VISIBLE_DEVICES="$gpu" \
    python src/carte_eval.py --dataset "$dataset" --numshot "$numshot"
}

# Create arrays of all experiments
datasets=(albert bank blood calhousing compas covertype credit_card_default creditg diabetes electricity eye_movements heart income jungle road_safety)
numshots=(8 16 32 64 128 256)

# Create array of all experiment combinations
experiments=()
for dataset in "${datasets[@]}"; do
    for numshot in "${numshots[@]}"; do
        experiments+=("$dataset $numshot")
    done
done

# Run experiments in parallel across 4 GPUs
gpu_count=4
for i in "${!experiments[@]}"; do
    gpu=$((i % gpu_count))
    experiment=(${experiments[i]})
    dataset=${experiment[0]}
    numshot=${experiment[1]}
    
    # Run in background
    run_experiment "$dataset" "$numshot" "$gpu" &
    
    # Limit the number of concurrent jobs to avoid overwhelming the system
    if (( (i + 1) % gpu_count == 0 )); then
        wait  # Wait for current batch to complete before starting next batch
    fi
done

# Wait for any remaining background jobs
wait
echo "All CARTE experiments completed!"