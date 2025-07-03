#!/bin/bash
set -x

# --- 1. 环境变量设置 ---
export PYTHONPATH=$(pwd):$PYTHONPATH
export NCCL_DEBUG=WARN
export WANDB_MODE=disabled
export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=true

# --- 2. 核心路径和名称变量设置 ---
PROJECT_NAME='QWEN2.5-EM-RL'
EXPERIMENT_NAME='em-rl-sequence-fsdp-2gpu-final-attempt'
DATA_PATH=./verl_math_dataset
SFT_MODEL_PATH=../../Qwen/Qwen2.5-Math-7B
CKPT_PATH=./checkpoints

# --- 3. 启动训练命令 ---
# ‼️ 关键：我们使用 python 而不是 torchrun 来启动。
python verl/trainer/main_ppo.py \
    \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.rollout.name=hf \
    data.max_prompt_length=256 \
    data.max_response_length=256 \
    data.train_batch_size=4 \
    data.val_batch_size=4 \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size=1 \
    verifier.type=default \
    verifier.reward_coef=0.0 \
    actor_rollout_ref.actor.strategy=fsdp \
    critic.strategy=fsdp \
    reward_model.strategy=fsdp \
    reward_model.enable=True \
    reward_model.rm_type=em_rl_sequence \
    reward_model.rm_coef=1.0 \
    algorithm.adv_estimator=rloo \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    data.train_files=["$DATA_PATH/train.parquet"] \
    data.val_files=["$DATA_PATH/validation.parquet"] \
    actor_rollout_ref.model.path=$SFT_MODEL_PATH \
    critic.model.path=$SFT_MODEL_PATH \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    trainer.logger="['console']" \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$CKPT_PATH/$PROJECT_NAME/$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    trainer.total_epochs=1 \
    data.n_samples=2