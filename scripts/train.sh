#!/bin/bash

DATASET=$1
GROUP_KEY=${2:-""}
DATASET_NAME=$3

if [[ $DATASET == "custom" ]];
then
    if [[ $GROUP_KEY == "tdc" ]];
    then
        if [[ $DATASET_NAME == "bbb_martins" ]];
        then
            python ./litgnn/train.py \
                dataset=$DATASET \
                dataset.group_key=$GROUP_KEY \
                dataset.dataset_name=$DATASET_NAME \
                ++train.dataset.split_sizes=[0.8,0.2,0.0] \
                ++train.epochs=50 \
                model=cmpnn \
                model.out_channels=1
        fi
    fi
elif [[ $DATASET == "molecule_net" ]];
then
    if [[ $DATASET_NAME == "bbb_martins" ]];
    then
        python ./litgnn/train.py \
            dataset=$DATASET \
            dataset.dataset_name=$DATASET_NAME \
            ++train.dataset.split=scaffold_split \
            ++train.dataset.split_sizes=[0.8,0.1,0.1] \
            ++train.epochs=50 \
            model=cmpnn \
            model.out_channels=1
    fi 
fi