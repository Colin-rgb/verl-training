#!/bin/bash

# 打印出脚本执行的每一行命令，非常有助于调试
set -x

# --- 1. 环境变量设置 (保持不变) ---
export NCCL_DEBUG=WARN
# 如果您使用W&B，请填入您的API Key
export WANDB_API_KEY='YOUR_WANDB_API_KEY' 
export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=true

# --- 2. 核心路径和名称变量设置 ---
# ‼️ 以下是您唯一需要手动修改的部分

PROJECT_NAME='QWEN2.5-EM-RL'
EXPERIMENT_NAME='em-rl-sequence-fsdp-single-gpu'

# ‼️ 指向您处理好的数据集所在的目录 (相对于 training/ 目录)
DATA_PATH=./verl_math_dataset 

# ‼️ 指向您的 qwen2.5-Math-7B 模型的存放路径
SFT_MODEL_PATH=../../Qwen/Qwen2.5-Math-7B

# ‼️ 指定一个目录，用于保存训练过程中生成的模型检查点 (checkpoints)
CKPT_PATH=./checkpoints

# --- 3. 启动verl训练命令 ---
# 使用 torchrun 启动，并指定使用1个GPU
torchrun --nproc_per_node=1 --master_port 29501 -m verl.trainer.main_ppo \
    \
    # --- 关键修改区域：在这里覆盖所有默认配置 ---
    \
    # 3.1 指定后端策略为 FSDP
    actor_rollout_ref.actor.strategy=fsdp \
    critic.strategy=fsdp \
    reward_model.strategy=fsdp \
    \
    # 3.2 EM-RL 核心算法配置
    reward_model.enable=True \
    reward_model.rm_type=em_rl_sequence \
    reward_model.rm_coef=1.0 \
    algorithm.adv_estimator=rloo \
    algorithm.kl_ctrl.kl_coef=0.001 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    \
    # 3.3 数据和模型路径配置
    data.train_files=["$DATA_PATH/train.parquet"] \
    data.val_files=["$DATA_PATH/validation.parquet"] \
    actor_rollout_ref.model.path=$SFT_MODEL_PATH \
    critic.model.path=$SFT_MODEL_PATH \
    \
    # 3.4 单GPU显存优化配置
    data.train_batch_size=8 \
    data.val_batch_size=8 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    \
    # 3.5 训练和日志配置
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$CKPT_PATH/$PROJECT_NAME/$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    trainer.total_epochs=1 \
    \
    # 3.6 论文中的其他相关参数
    data.n_samples=4