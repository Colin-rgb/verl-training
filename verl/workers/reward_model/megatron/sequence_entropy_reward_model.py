import torch
from tensordict import TensorDict
from verl import DataProto
from .reward_model import MegatronRewardModel # 导入基类

class MegatronSequenceEntropyRewardModel(MegatronRewardModel):
    """
    专门用于计算 EM-RL-sequence 奖励的模型。
    奖励直接源于策略模型（Actor）在Rollout时计算的对数概率，因此本类不执行任何模型前向传播。
    """
    def __init__(self, *args, **kwargs):
        # 显式地调用父类构造函数，即使在此实现中很多参数用不到
        super().__init__(*args, **kwargs)
        print("INFO: Initialized MegatronSequenceEntropyRewardModel.")
        print("      This model computes reward from 'old_log_probs' and does not perform forward passes.")

    @torch.no_grad()
    def compute_reward(self, data: DataProto) -> DataProto:
        """
        重写此方法以实现 EM-RL-sequence 奖励。
        奖励值为序列的总对数概率：r_traj(y) = log π_θ(y)。
        """
        # 1. 从 DataProto 中获取 Actor 计算好的序列各 token 的对数概率
        log_probs = data.batch['old_log_probs']
        
        # 2. 获取响应部分的掩码，用于忽略填充（padding）部分
        prompt_length = data.batch['prompts'].shape[-1]
        response_mask = data.batch['attention_mask'][:, prompt_length:]

        # 3. 计算每个序列的总对数概率作为“轨迹奖励”
        sequence_log_prob = (log_probs * response_mask).sum(dim=1)

        # 4. 创建一个 token-level 的零张量，用于承载奖励
        token_level_rewards = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        
        # 5. 将序列级奖励赋给每个序列的最后一个有效 token
        last_token_indices = response_mask.sum(dim=1).long() - 1
        
        token_level_rewards.scatter_(
            1, 
            last_token_indices.unsqueeze(-1), 
            sequence_log_prob.unsqueeze(-1)
        )
        
        # 6. 将结果封装在 DataProto 中返回
        batch = TensorDict({'rm_scores': token_level_rewards}, batch_size=log_probs.shape[0])

        return DataProto(batch=batch)