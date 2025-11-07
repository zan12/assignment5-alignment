import argparse
import time
import torch
import wandb

import torch.nn.functional as F
from collections import Counter
from typing import Any, Iterator

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase
from torch.optim import AdamW
from vllm import LLM, SamplingParams

from .drgrpo_grader import r1_zero_reward_fn
from .eval_math import get_jsonl_data_stream, apply_prompt_template, make_math_eval_data, evaluate_vllm
from .init_vllm import init_vllm, load_policy_into_vllm_instance


ALL_CORRECT = "{'format_reward': 1.0, 'answer_reward': 1.0, 'reward': 1.0}"
FORMAT_CORRECT = "{'format_reward': 1.0, 'answer_reward': 0.0, 'reward': 0.0}"
NO_HIT = "{'format_reward': 0.0, 'answer_reward': 0.0, 'reward': 0.0}"


def make_config(config, argv):
    parser = argparse.ArgumentParser()
    for k, v in config.items():
        if k in ["num_epochs", "batch_size", "gradient_accumulation_steps"]:
            parser.add_argument(f"--{k}", dest=k, default=v, type=int)
        elif k in ["lr"]:
            parser.add_argument(f"--{k}", dest=k, default=v, type=float)
        else:
            parser.add_argument(f"--{k}", dest=k, default=v)
    config = parser.parse_args(argv)
    return config


def batch(lines: Iterator[dict[str, Any]], batch_size: int) -> Iterator:
    buffer = []
    for line in lines:
        buffer.append((line["prompt"], line["response"]))
        if len(buffer) == batch_size:
            yield list(zip(*buffer))
            buffer = []


def tokenize_prompt_and_output(prompt_strs: list[str], output_strs: list[str], tokenizer: PreTrainedTokenizerBase):
    """Tokenize the prompt and output strings, and construct a mask that is 1 for the response tokens and 0 for other tokens (prompt or padding). 
    
    Returns:
        dict[str, torch.Tensor]:
            "input_ids": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                the tokenized prompt and output strings, with the final token sliced off.
            "labels": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                shifted input_ids (i.e., the input_ids without the first token).
            "response_mask": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                a mask on the response tokens in `labels`.
    
    An Example:
        input_ids: xyzabcpp
        labels:    yzabcppp
        mask:      00111000
    """
    batch_size = len(list(prompt_strs))
    assert len(prompt_strs) == len(output_strs), "Number of prompts and outputs should be the same."

    prompt_ids = [tokenizer.encode(p) for p in prompt_strs]
    output_ids = [tokenizer.encode(p) for p in output_strs]
    prompt_len = [len(i) for i in prompt_ids]
    output_len = [len(i) for i in output_ids]
    max_prompt_output_len = max(map(sum, zip(prompt_len, output_len)))
    assert max_prompt_output_len > 1, "Prompt and output should not be empty."
    input_ids = torch.fill(
        torch.zeros((batch_size, max_prompt_output_len-1), dtype=torch.int32),
        tokenizer.pad_token_id,
    )
    labels = torch.fill(
        torch.zeros((batch_size, max_prompt_output_len-1), dtype=torch.int64),
        tokenizer.pad_token_id,
    )
    response_mask = torch.zeros((batch_size, max_prompt_output_len-1), dtype=torch.int32)
    
    for i in range(batch_size):
        all_ids = torch.tensor(prompt_ids[i] + output_ids[i])
        prompt_len, seq_len = len(prompt_ids[i]), all_ids.shape[-1]
        if seq_len < max_prompt_output_len:
            input_ids[i, :seq_len] = all_ids
        else:
            input_ids[i, :seq_len-1] = all_ids[:-1]
        labels[i, :seq_len-1] = all_ids[1:]
        response_mask[i, prompt_len-1:seq_len-1] = 1

    output = dict(
        input_ids = input_ids,
        labels = labels,
        response_mask = response_mask,
    )
    return output


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Get the entropy of the next-token predictions."""
    # assume vocab is the last dimension
    logits -= torch.max(logits, dim=-1, keepdim=True)[0]
    log_prob = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    return torch.sum(-torch.exp(log_prob)*log_prob, dim=-1)


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    logits = model(input_ids).logits
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs = torch.gather(log_probs, -1, labels[..., None]).squeeze(-1)
    output = dict(log_probs = log_probs)

    if return_token_entropy:
        entropy = compute_entropy(logits)
        output["token_entropy"] = entropy
    return output


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> torch.Tensor:
    masked_tensor = mask*tensor
    summed_tensor = torch.sum(masked_tensor, dim=dim)
    return summed_tensor / normalize_constant


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Execute a forward-and-backward pass on a microbatch."""
    batch_size = policy_log_probs.shape[0]
    loss = -masked_normalize(policy_log_probs, response_mask, dim=None, normalize_constant=normalize_constant) / batch_size / gradient_accumulation_steps
    loss.backward()
    metadata = None
    return loss, metadata


def log_generations(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, eval_dir: str, device: torch.device):
    all_rewards = []
    for jline in get_jsonl_data_stream(eval_dir):
        prompt = apply_prompt_template(jline["problem"], "./cs336_alignment/prompts/r1_zero.prompt")
        gt = jline["solution"]
        # prompt = "What is 4 devided by 2?"
        # gt = "The answer is 2."

        # Get input and output ids    
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        output_ids = model.generate(input_ids)
        input_size, output_size = input_ids.numel(), output_ids[0].numel()
        output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # print("Output:", output)

        # Rewards
        reward_fn = r1_zero_reward_fn
        rewards = reward_fn(output, gt)
        all_rewards.append(str(rewards))

        # print("Rewards:", rewards)

        # Average response entropy
        # entropy = torch.mean(
        #     compute_entropy(model(output_ids).logits[:, input_size-1:output_size-1, :])
        # )
        # print("Entropy:", entropy)
    print(Counter(all_rewards))


def calc_entropy(model, tokenizer, prompts, generations):
    all_entropy = []
    with torch.no_grad():
        for p,g in zip(prompts, generations):
            try:
                input_ids = tokenizer(p, return_tensors="pt").input_ids.to(model.device)
                output_ids = tokenizer(g, return_tensors="pt").input_ids.to(model.device)
                text_ids = torch.concat([input_ids, output_ids], dim=-1)
                input_size, total_size = input_ids.numel(), text_ids.numel()
                entropy = torch.mean(
                    compute_entropy(model(text_ids).logits[:, input_size-1:total_size-1, :])
                )
                all_entropy.append(entropy.item())
            except Exception as e:
                print(e)
    return sum(all_entropy) / len(all_entropy)


def run(argv=None):
    config = dict(
        model_dir = "./data/a5-alignment/models/Qwen2.5-Math-1.5B",
        tokenizer_dir = "./data/a5-alignment/models/Qwen2.5-Math-1.5B",
        input_dir = "./data/a5-alignment/MATH/r1_distilled_train_stopped_shuf_128.jsonl",
        eval_dir = "./data/a5-alignment/MATH/validation.jsonl",
        prompt_template_dir = "./cs336_alignment/prompts/r1_zero.prompt",
        num_epochs = 5,
        batch_size = 2,
        gradient_accumulation_steps = 16,
        lr = 1e-4,
    )
    config = make_config(config, argv)
    model_dir = config.model_dir
    tokenizer_dir = config.tokenizer_dir
    input_dir = config.input_dir
    eval_dir = config.eval_dir
    prompt_template_dir = config.prompt_template_dir
    num_epochs = config.num_epochs
    batch_size = config.batch_size
    gradient_accumulation_steps = config.gradient_accumulation_steps
    lr = config.lr

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vllm_device = "cuda:7"
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(device)
    vllm_model = init_vllm(model_id=model_dir, device=vllm_device, seed=43)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    opt = AdamW(model.parameters(), lr=lr)
    input_shortname = input_dir.split("r1_distilled_train_stopped_")[-1].split(".jsonl")[0]

    output_dir = f"./ckpts/sft/math_longcot_b{batch_size*gradient_accumulation_steps}_lr{lr}_ep{num_epochs}_dat_{input_shortname}"
    cum_loss, cum_token, current_time = 0, 0, time.time()
    wandb.init(project="cs336", name=f"{current_time}-assignment5-alignment", config=config)
    for t, train_batch in enumerate(batch(get_jsonl_data_stream(input_dir, num_epochs), batch_size)):
        train_batch = tokenize_prompt_and_output(*train_batch, tokenizer)
        input_ids = train_batch["input_ids"].to(device)
        labels = train_batch["labels"].to(device)
        response_mask = train_batch["response_mask"].to(device)
        policy_log_probs = get_response_log_probs(model, input_ids, labels)["log_probs"]
        loss, _ = sft_microbatch_train_step(policy_log_probs, response_mask, 
        gradient_accumulation_steps, normalize_constant=torch.sum(response_mask) / batch_size)
        cum_loss += loss * torch.sum(response_mask) * gradient_accumulation_steps
        cum_token += torch.sum(response_mask)
        
        if (t+1) % gradient_accumulation_steps == 0:
            print("per-token loss:", cum_loss/cum_token)
            updated_time = time.time()
            wandb.log({
                "step": (t+1) // gradient_accumulation_steps,
                "step_time": updated_time - current_time,
                "loss": cum_loss/cum_token,
            })
            cum_loss, cum_token, current_time = 0, 0, time.time()
            opt.step()
            opt.zero_grad()
    model.save_pretrained(save_directory=output_dir)
    tokenizer.save_pretrained(save_directory=output_dir)

    load_policy_into_vllm_instance(model, vllm_model)

    prompts, ground_truths = make_math_eval_data(
        eval_dir, prompt_template_dir
    )
    all_summarize = evaluate_vllm(
        vllm_model=vllm_model,
        reward_fn=r1_zero_reward_fn,
        prompts=prompts,
        ground_truths=ground_truths,
        eval_sampling_params=SamplingParams(
            temperature=1.,
            top_p=1.,
            min_tokens=4,
            max_tokens=1024,
            stop=["</answer>"],
            include_stop_str_in_output=True,
        ),
    )
    generations = [s["generation"] for s in all_summarize]
    rewards = Counter([str(s["rewards"]) for s in all_summarize])
    all_correct, format_correct, no_hit = rewards[ALL_CORRECT], rewards[FORMAT_CORRECT], rewards[NO_HIT]
    total_val = all_correct + format_correct + no_hit
    entropy = calc_entropy(model, tokenizer, prompts, generations)
    wandb.log({
        "eval_all_correct": all_correct/total_val,
        "eval_format_correct": format_correct/total_val,
        "eval_no_hit": no_hit/total_val,
        "eval_entropy": entropy,
    })
    # log_generations(model, tokenizer, eval_dir, device)


if __name__ == "__main__":
    run()
