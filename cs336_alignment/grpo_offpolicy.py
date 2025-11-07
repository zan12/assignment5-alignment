import argparse
import random
import time
import torch
import wandb

from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Literal
from vllm import SamplingParams

from .drgrpo_grader import r1_zero_reward_fn
from .eval_math import evaluate_vllm, make_math_eval_data
from .expert import eval_expert
from .grpo import compute_group_normalized_rewards, grpo_microbatch_train_step
from .init_vllm import init_vllm, load_policy_into_vllm_instance
from .sft import get_response_log_probs, tokenize_prompt_and_output


def get_grpo_train_step_data(train_data, st, tb):
    input_ids, labels, response_mask, old_log_probs, advantages, raw_rewards = train_data
    return input_ids[st*tb:(st+1)*tb,...], labels[st*tb:(st+1)*tb, ...], response_mask[st*tb:(st+1)*tb, ...], old_log_probs[st*tb:(st+1)*tb, ...], advantages[st*tb:(st+1)*tb, ...], raw_rewards[st*tb:(st+1)*tb, ...]
    

def rollout_grpo_offpolicy(
    vllm_model,
    model,
    tokenizer,
    prompts,
    gts,
    rollout_batch_size,
    group_size,
    advantage_eps,
    normalize_by_std,
    device,
):
    rollout_per_group = rollout_batch_size//group_size
    grpo_batch = random.sample(list(zip(prompts, gts)), rollout_per_group)
    prompt, gt = zip(*grpo_batch)
    sampling_params = SamplingParams(n=group_size, temperature=1., top_p=1., min_tokens=4, max_tokens=1024, stop=["</answer>"], include_stop_str_in_output=True,)
    all_summarize = evaluate_vllm(vllm_model, prompt, gt, r1_zero_reward_fn, sampling_params)
    rollout_prompts = [s["prompt"] for s in all_summarize]
    rollout_responses = [s["generation"] for s in all_summarize]
    rollout_gt = [s["gt"] for s in all_summarize]
    # Compute advantages raw rewards and advantages
    advantages, raw_rewards, _ = compute_group_normalized_rewards(r1_zero_reward_fn, rollout_responses, rollout_gt, group_size, advantage_eps, normalize_by_std)
    tb = tokenize_prompt_and_output(rollout_prompts, rollout_responses, tokenizer)
    input_ids, labels, response_mask = tb["input_ids"].to(device), tb["labels"].to(device), tb["response_mask"].to(device)
    old_log_probs = []
    for i in range(rollout_batch_size//2):
        old_log_probs.append(get_response_log_probs(model, input_ids[i*2:(i+1)*2,:], labels[i*2:(i+1)*2,:])["log_probs"])
    old_log_probs = torch.cat(old_log_probs, dim=0)
    return input_ids, labels, response_mask, old_log_probs, torch.tensor(advantages)[...,None].to(device), torch.tensor(raw_rewards)[...,None].to(device)


def make_config(config, argv):
    parser = argparse.ArgumentParser()
    for k,v in config.items():
        if k in ["n_grpo_steps", "rollout_batch_size", "group_size", "gradient_accumulation_steps"]:
            parser.add_argument(f"--{k}", dest=k, default=v, type=int)
        elif k in ["lr", "cliprange", "advantage_eps"]:
            parser.add_argument(f"--{k}", dest=k, default=v, type=float)
        elif k in ["use_std_normalization"]:
            parser.add_argument(f"--{k}", action="store_true")
        elif k == "loss_type":
            parser.add_argument(f"--{k}", choices=["no_baseline", "reinforce_with_baseline", "grpo_clip"], default=v)
        else:
            parser.add_argument(f"--{k}", dest=k, default=v)
    return parser.parse_args(argv)


def run(argv=None):
    config = dict(
        model_dir = "./data/a5-alignment/models/Qwen2.5-Math-1.5B",
        tokenizer_dir = "./data/a5-alignment/models/Qwen2.5-Math-1.5B",
        input_dir = "./data/a5-alignment/MATH/train.jsonl",
        output_dir = "./data/a5-alignment/grpo/offpolicy",
        eval_dir = "./data/a5-alignment/MATH/validation.jsonl",
        prompt_template_dir = "./cs336_alignment/prompts/r1_zero.prompt",
        loss_type = "reinforce_with_baseline",
        n_grpo_steps = 200,
        lr = 1e-5,
        advantage_eps = 1e-6,
        rollout_batch_size = 256,
        group_size = 8,
        train_batch_size = 2,
        gradient_accumulation_steps = 128,
        use_std_normalization = True,
        cliprange = 0.2,
    )
    config = make_config(config, argv)
    model_dir = config.model_dir
    tokenizer_dir = config.tokenizer_dir
    input_dir = config.input_dir
    output_dir = config.output_dir
    eval_dir = config.eval_dir
    prompt_template_dir = config.prompt_template_dir
    loss_type = config.loss_type
    n_grpo_steps = config.n_grpo_steps
    lr = config.lr
    advantage_eps = config.advantage_eps
    rollout_batch_size = config.rollout_batch_size
    group_size = config.group_size
    train_batch_size = config.train_batch_size
    gradient_accumulation_steps = config.gradient_accumulation_steps
    use_std_normalization = config.use_std_normalization
    cliprange = config.cliprange
    
    device, vllm_device = "cuda", "cuda:7"
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(device)
    vllm_model = init_vllm(model_id=model_dir, device=vllm_device, seed=43)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=0.0, betas=(0.9,0.95))
    
    prompts, gts = make_math_eval_data(input_dir, prompt_template_dir)
    current_time = time.time()
    wandb.init(project="cs336", name=f"{current_time}-assignment5-alignment", config=config, tags=["grpo", "baseline", loss_type])
    for t in range(n_grpo_steps):
        print(f"GRPO Iteration {t}")
        load_policy_into_vllm_instance(model, vllm_model)
        with torch.no_grad():
            train_data = rollout_grpo_offpolicy(vllm_model, model, tokenizer, prompts, gts, rollout_batch_size, group_size, advantage_eps, use_std_normalization, device)
        for st in range(rollout_batch_size//train_batch_size):
            input_ids, labels, response_mask, old_log_probs, advantages, raw_rewards = get_grpo_train_step_data(train_data, st, train_batch_size)
            policy_log_probs = get_response_log_probs(model, input_ids, labels)["log_probs"]
            grpo_microbatch_train_step(policy_log_probs, response_mask, gradient_accumulation_steps, loss_type, raw_rewards, advantages, old_log_probs, cliprange)
            if (st+1) % gradient_accumulation_steps == 0:
                opt.step()
                opt.zero_grad()
        if (t+1) % 10 == 0:
            all_correct, format_correct, no_hit, entropy = eval_expert(vllm_model, model, tokenizer, eval_dir, prompt_template_dir)
            wandb.log({
                "step": t+1,
                "all_correct": all_correct/(all_correct+format_correct+no_hit),
                "format_correct": format_correct/(all_correct+format_correct+no_hit),
                "no_hit": no_hit/(all_correct+format_correct+no_hit),
                "entropy": entropy,
            })
                

if __name__ == "__main__":
    run()