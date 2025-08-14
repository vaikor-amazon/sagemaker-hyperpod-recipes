#!/bin/bash

# Original Copyright (c), NVIDIA CORPORATION. Modifications © Amazon.com

#Users should setup their cluster type in /recipes_collection/config.yaml

SAGEMAKER_TRAINING_LAUNCHER_DIR=${SAGEMAKER_TRAINING_LAUNCHER_DIR:-"$(pwd)"}

HF_MODEL_NAME_OR_PATH="${HF_MODEL_NAME_OR_PATH}" # HuggingFace pretrained model name or path
HF_ACCESS_TOKEN="${HF_ACCESS_TOKEN}" # Optional HuggingFace access token

TRAIN_DIR="${TRAIN_DIR}" # Location of training dataset
VAL_DIR="${VAL_DIR}" # Location of validation dataset

EXP_DIR="${EXP_DIR}" # Location to save experiment info including logging, checkpoints, ect


HYDRA_FULL_ERROR=1 python3 "${SAGEMAKER_TRAINING_LAUNCHER_DIR}/main.py" \
    recipes=fine-tuning/gpt_oss/hf_gpt_oss_120b_seq4k_gpu_lora \
    container="658645717510.dkr.ecr.us-west-2.amazonaws.com/smdistributed-modelparallel:sm-pytorch_gpt_oss_patch_pt-2.7_cuda12.8" \
    base_results_dir="${SAGEMAKER_TRAINING_LAUNCHER_DIR}/results" \
    recipes.run.name="hf-gpt-oss-120b-lora" \
    recipes.exp_manager.exp_dir="$EXP_DIR" \
    recipes.trainer.num_nodes=1 \
    recipes.model.data.train_dir="$TRAIN_DIR" \
    recipes.model.data.val_dir="$VAL_DIR" \
    recipes.model.hf_model_name_or_path="$HF_MODEL_NAME_OR_PATH" \
    recipes.model.hf_access_token="$HF_ACCESS_TOKEN" \
