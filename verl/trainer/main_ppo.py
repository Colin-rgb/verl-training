# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""
import json
import os
import statistics
import warnings
from functools import partial

from verl import DataProto
import torch
from verl.utils.reward_score import gsm8k, math
from verl.trainer.ppo.ray_trainer import RayPRIMETrainer


def _select_rm_score_fn(data_source):
    if data_source == 'openai/gsm8k':
        return gsm8k.compute_score
    elif data_source == 'lighteval/MATH':
        return math.compute_score
    else:
        raise NotImplementedError


class RewardManager():
    """The reward manager.
    """
    def __init__(self, tokenizer, num_examine, config) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.config = config
        
        # ==================== 关键修复：按需导入 ====================
        # 我们将这里的无条件导入，修改为按需导入，以修复 'pyext' 模块找不到的错误。
        
        verifier_name = self.config.verifier.get('type', 'default')
        
        # 只有当 verifier 类型真的是 'prime' 时，才执行导入操作。
        if verifier_name == 'prime': 
            print("INFO: 'prime' verifier is configured. Importing prime-specific modules.")
            from verl.utils.reward_score.prime import compute_score
            self.verifier_func = compute_score
        # 当设置为 'default' 或其他任何非 'prime' 类型时，都不会触发导入，从而避免错误。
        elif verifier_name == 'default':
            print("INFO: 'default' verifier is configured. The verifier function will not be used.")
            self.verifier_func = None 
        else:
            raise NotImplementedError(f"Verifier type '{verifier_name}' is not implemented.")
        # ===================== 修复结束 =====================

    def verify(self, data):
        # 如果 verifier_func 未被初始化（例如，当 type 为 'default' 时），则直接返回，不进行验证。
        if not self.verifier_func:
            warnings.warn("Verifier function is not available. Skipping verification.")
            return [], {}
            
        response_ids = data.batch['responses']
        response_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        ground_truth = [data_item.non_tensor_batch['reward_model']['ground_truth'] for data_item in data]
        score = self.verifier_func(completions=response_str, references=ground_truth,
                                   tasks=data.non_tensor_batch['ability'])
        data.batch['acc'] = torch.tensor(score, dtype=torch.float32, device=data.batch['responses'].device)
        reward_metrics = {}
        for ability in list(set(data.non_tensor_batch['ability'])):
            score_ = [data.batch['acc'][i].item() for i in range(len(data.batch['acc'])) if
                      data.non_tensor_batch['ability'][i] == ability]
            reward_metrics[f'{ability}'] = statistics.mean(score_)
        reward_metrics['all'] = data.batch['acc'].mean().item()

        for i,response_str_ in enumerate(response_str):
            if i>=self.num_examine:
                break
            example = data.batch[i]['input_ids']
            print(self.tokenizer.decode(example, skip_special_tokens=True))

        return score, reward_metrics

    def __call__(self, data: DataProto):
        # 奖励聚合逻辑保持我们之前重构后的版本，它已经很健壮了。
        reward_tensor_dict = {}
        reward_metrics = {}
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        if self.config.verifier.reward_coef != 0 and self.verifier_func is not None:
            verifier_reward = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
            prompt_length = data.batch['prompts'].shape[-1]
            valid_response_length = data.batch['attention_mask'][:, prompt_length:].sum(-1)
            
            if 'acc' in data.batch:
                verifier_score = data.batch['acc'].cpu().numpy().tolist()
            else:
                verifier_score, verifier_metrics_update = self.verify(data)
                reward_metrics.update(verifier_metrics_update)
            
            for i in range(verifier_reward.shape[0]):
                verifier_reward[i, valid_response_length[i] - 1] += verifier_score[i]

            reward_tensor_dict['verifier_scores'] = verifier_reward
            reward_tensor += self.config.verifier.reward_coef * verifier_reward
            reward_metrics['reward/verifier'] = verifier_reward.sum(dim=1).mean().item()

        rm_type = self.config.reward_model.get('rm_type', 'normal')

        if 'rm_scores' in data.batch.keys() and self.config.reward_model.rm_coef != 0:
            if rm_type == 'em_rl_sequence':
                em_sequence_reward = data.batch['rm_scores']
                reward_tensor_dict['em_rl_sequence_scores'] = em_sequence_reward
                reward_tensor += self.config.reward_model.rm_coef * em_sequence_reward
                reward_metrics['reward/em_rl_sequence'] = em_sequence_reward.sum(dim=1).mean().item()
            elif rm_type in ['normal', 'prime']:
                model_based_reward = data.batch['rm_scores']
                reward_tensor_dict['model_based_scores'] = model_based_reward
                reward_tensor += self.config.reward_model.rm_coef * model_based_reward
                reward_metrics['reward/model_based'] = model_based_reward.sum(dim=1).mean().item()
            else:
                warnings.warn(f"Warning: 'rm_scores' present but rm_type '{rm_type}' is not explicitly handled.")
                reward_tensor += self.config.reward_model.rm_coef * data.batch['rm_scores']

        reward_tensor_dict['all'] = reward_tensor
        reward_metrics['reward/final_total'] = reward_tensor.sum(dim=-1).mean().item()

        return reward_tensor_dict, reward_metrics


import ray
import hydra

@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        if os.path.isfile(str(config.trainer.runtime_env)):
            with open(str(config.trainer.runtime_env), 'r') as f:
                runtime_env = json.load(f)
            ray.init(runtime_env=runtime_env)
        else:
            ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config))

# @ray.remote
# def main_task(config):
#     from verl.utils.fs import copy_local_path_from_hdfs
#     from transformers import AutoTokenizer

#     from pprint import pprint
#     from omegaconf import OmegaConf
#     pprint(OmegaConf.to_container(config, resolve=True))
#     OmegaConf.resolve(config)

#     local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

#     from verl.utils import hf_tokenizer
#     tokenizer = hf_tokenizer(local_path)

#     if config.actor_rollout_ref.actor.strategy == 'fsdp':
#         from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
#         from verl.single_controller.ray import RayWorkerGroup
#         ray_worker_group_cls = RayWorkerGroup
#     elif config.actor_rollout_ref.actor.strategy == 'megatron':
#         from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
#         from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
#         ray_worker_group_cls = NVMegatronRayWorkerGroup
#     else:
#         raise NotImplementedError

#     from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

#     role_worker_mapping = {
#         Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
#         Role.Critic: ray.remote(CriticWorker),
#         Role.RefPolicy: ray.remote(ActorRolloutRefWorker)
#     }

#     global_pool_id = 'global_pool'
#     resource_pool_spec = {
#         global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
#     }
#     mapping = {
#         Role.ActorRollout: global_pool_id,
#         Role.Critic: global_pool_id,
#         Role.RefPolicy: global_pool_id,
#     }

#     # Worker的选择逻辑保持我们之前的版本即可，它已经很完善
#     if config.reward_model.enable and config.reward_model.rm_coef != 0.:
#         rm_type = config.reward_model.get('rm_type', 'normal')

#         if rm_type == 'prime':
#             from verl.workers.fsdp_workers import PRIMERewardModelWorker
#             role_worker_mapping[Role.RewardModel] = ray.remote(PRIMERewardModelWorker)
#         else: # For 'normal' and 'em_rl_sequence'
#             if config.reward_model.strategy == 'fsdp':
#                 from verl.workers.fsdp_workers import RewardModelWorker
#                 role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
#             elif config.reward_model.strategy == 'megatron':
#                 from verl.workers.megatron_workers import RewardModelWorker
#                 role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
#             else:
#                 raise NotImplementedError(f"Strategy '{config.reward_model.strategy}' not supported.")
        
#         mapping[Role.RewardModel] = global_pool_id

# ==================== ‼️ 关键修改区域 ====================
@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker, RewardModelWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup
    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker, RewardModelWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup
    else:
        raise NotImplementedError

    # # --- 关闭KL散度计算版本 ---    
    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    # ==================== ‼️ 关键修改：按需初始化所有Worker ====================
    
    # 1. 总是初始化 Actor，它是必须的
    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
    }

    # 2. 只有在算法需要时 (非rloo)，才初始化 Critic
    if config.algorithm.adv_estimator != 'rloo':
        print("INFO: GAE or other value-based estimator detected. Initializing CriticWorker.")
        role_worker_mapping[Role.Critic] = ray.remote(CriticWorker)
    else:
        print("INFO: RLOO estimator detected. CriticWorker will NOT be initialized, saving resources.")

    # 3. 只有在KL系数大于0时，才初始化 RefPolicy
    if config.algorithm.kl_ctrl.kl_coef > 0:
        print(f"INFO: kl_coef ({config.algorithm.kl_ctrl.kl_coef}) > 0. Initializing RefPolicy worker.")
        role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
    else:
        print("INFO: kl_coef is 0. RefPolicy worker will NOT be initialized, saving resources.")

    # 4. 如果启用了奖励模型，添加 RewardModelWorker
    if config.reward_model.enable and config.reward_model.rm_coef != 0.:
        if config.reward_model.rm_type == 'prime':
            from verl.workers.fsdp_workers import PRIMERewardModelWorker
            role_worker_mapping[Role.RewardModel] = ray.remote(PRIMERewardModelWorker)
        else: # For 'normal' and our 'em_rl_sequence'
             role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
    
    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    
    mapping = {role: global_pool_id for role in role_worker_mapping.keys()}
    # ============================ 修改结束 ============================


    reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0, config=config)
    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1, config=config)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    trainer = RayPRIMETrainer(config=config,
                              tokenizer=tokenizer,
                              role_worker_mapping=role_worker_mapping,
                              resource_pool_manager=resource_pool_manager,
                              ray_worker_group_cls=ray_worker_group_cls,
                              reward_fn=reward_fn,
                              val_reward_fn=val_reward_fn)
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()