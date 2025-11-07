import itertools
import numpy as np
import torch

from typing import Callable, Literal


def compute_group_normalized_rewards(
    reward_fn: Callable,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float = 1e-2,
    normalize_by_std: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    assert len(repeated_ground_truths) == len(rollout_responses)
    response_gt = zip(rollout_responses, repeated_ground_truths)
    raw_rewards = [reward_fn(s,g)["reward"] for s,g in response_gt]
    raw_rewards_np = np.array(raw_rewards).reshape((-1,group_size))
    advantages = raw_rewards_np - raw_rewards_np.mean(axis=-1, keepdims=True)
    if normalize_by_std:
        advantages /= raw_rewards_np.std(axis=-1, keepdims=True, ddof=1) + advantage_eps
    advantages = list(np.reshape(advantages, (-1,)))
    metadata = {}
    return advantages, raw_rewards, metadata


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    return -raw_rewards_or_advantages * policy_log_probs


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    prob_ratio = torch.exp(policy_log_probs - old_log_probs)
    wo_clip_term = prob_ratio * advantages
    clip_term = torch.clip(prob_ratio, 1-cliprange, 1+cliprange) * advantages
    metadata = {}
    return -torch.minimum(wo_clip_term, clip_term), metadata


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor,
    advantages: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if loss_type == "no_baseline":
        return compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs), {}
    elif loss_type == "reinforce_with_baseline":
        return compute_naive_policy_gradient_loss(advantages, policy_log_probs), {}
    elif loss_type == "grpo_clip":
        return compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
    else:
        raise NotImplementedError
    

def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int | None = None):
    masked_tensor = tensor * mask
    return masked_tensor.sum(dim=dim) / mask.sum(dim=dim)


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss, metadata = compute_policy_gradient_loss(policy_log_probs, loss_type, raw_rewards, advantages, old_log_probs, cliprange)
    loss = masked_mean(loss, response_mask) / gradient_accumulation_steps
    loss.backward()
    return loss, metadata