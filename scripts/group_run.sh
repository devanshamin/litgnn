#!/bin/bash

# Function to check if the dataset group key is valid
check_dataset_group_key() {
    local DATASET_GROUP_KEY=$1
    if [[ ! "tdc molecule_net biogen" =~ $DATASET_GROUP_KEY ]]; then
        echo "Error: Invalid dataset group key '$DATASET_GROUP_KEY'. Available group keys are: tdc, molecule_net, biogen."
        exit 1
    fi
}

# Function to check if the run type is valid
check_run_type() {
    local RUN_TYPE=$1
    if [[ ! "standard hpo" =~ $RUN_TYPE ]]; then
        echo "Error: Invalid run type '$RUN_TYPE'. Available run types are: standard, hpo."
        exit 1
    fi
}

# Function to set the dataset split sizes based on the dataset group key
set_dataset_split_sizes() {
    local DATASET_GROUP_KEY=$1
    if [[ "tdc biogen" =~ $DATASET_GROUP_KEY ]]; then
        # Both the datasets provide separate train_val and test splits
        SPLIT_SIZES=[0.8,0.2,0.0]
    else
        SPLIT_SIZES=[0.8,0.1,0.1]
    fi
}

# Function to start the HPO run
start_hpo_run() {
    read -p "Enter no. of trials (default: 50): " N_TRIALS
    N_TRIALS=${N_TRIALS:-50}

    read -p "Enter no. of jobs (default: 1): " N_JOBS
    N_JOBS=${N_JOBS:-1}

    echo "Running optuna hyperparameter optimization on dataset: $DATASET_NAME"

    python ./litgnn/train.py \
        model=$MODEL \
        dataset=$DATASET_GROUP_KEY/$DATASET_NAME \
        ++train.dataset.split_sizes=$SPLIT_SIZES \
        +hpo/optuna=default \
        hpo/optuna/search_spaces@hydra.sweeper.params=$MODEL \
        ++hydra.sweeper.n_trials=$N_TRIALS \
        ++hydra.sweeper.n_jobs=$N_JOBS \
        -m
}

# Main script
MODEL=$1
DATASET_GROUP_KEY=$2
RUN_TYPE=${3:-standard}

check_dataset_group_key "$DATASET_GROUP_KEY"
check_run_type "$RUN_TYPE"
set_dataset_split_sizes "$DATASET_GROUP_KEY"

# Iterate through all dataset *.yaml files 
dir_path="litgnn/conf/dataset/$DATASET_GROUP_KEY"
for file_path in "$dir_path"/*.yaml; do
    DATASET_NAME=$(basename "$file_path" .yaml)
    if [[ $RUN_TYPE == "hpo" ]]; then
        start_hpo_run
    else
        echo "Training on dataset: $DATASET_NAME"
        python ./litgnn/train.py \
            model=$MODEL \
            dataset=$DATASET_GROUP_KEY/$DATASET_NAME \
            ++train.dataset.split_sizes=$SPLIT_SIZES
    fi
done