import json
import os
import time
import torch

from collections import Counter
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

from .eval_math import get_jsonl_data_stream, make_math_eval_data, evaluate_vllm
from .init_vllm import init_vllm, load_policy_into_vllm_instance
from .sft import batch, calc_entropy, get_response_log_probs, make_config, sft_microbatch_train_step, tokenize_prompt_and_output, ALL_CORRECT, FORMAT_CORRECT, NO_HIT


def get_expert_training_data(
    vllm_model,
    model,
    tokenizer,
    input_dir,
    prompt_template_dir,
    output_dir,
    expert_iter,
):
    prompts, ground_truths = make_math_eval_data(
        input_dir, prompt_template_dir
    )
    all_summarize = evaluate_vllm(
        vllm_model=vllm_model,
        prompts=prompts,
        ground_truths=ground_truths,
    )
    training_prompts, training_generations = [], []
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, f"expert_iter_{expert_iter}.jsonl")
    correct, total = 0, 0
    with open(output_file, "w") as f:
        for s in all_summarize:
            total += 1
            if s["rewards"]["reward"] == 1.0:
                json.dump({
                    "prompt": s["prompt"],
                    "response": s["generation"],
                }, f)
                training_prompts.append(s["prompt"])
                training_generations.append(s["generation"])
                f.write("\n")
                correct += 1
    entropy = calc_entropy(model, tokenizer, training_prompts, training_generations)
    print(f"Training Correct: {correct} ({100*correct/total}%)")
    print(f"Training Entropy: {entropy}")
    return output_file


def eval_expert(
    vllm_model, 
    model, 
    tokenizer, 
    eval_dir, 
    prompt_template_dir
):
    prompts, ground_truths = make_math_eval_data(
        eval_dir, prompt_template_dir
    )
    all_summarize = evaluate_vllm(
        vllm_model=vllm_model,
        prompts=prompts,
        ground_truths=ground_truths,
    )
    generations = [s["generation"] for s in all_summarize]
    rewards = Counter([str(s["rewards"]) for s in all_summarize])
    all_correct, format_correct, no_hit = rewards[ALL_CORRECT], rewards[FORMAT_CORRECT], rewards[NO_HIT]
    total_val = all_correct + format_correct + no_hit
    entropy = calc_entropy(model, tokenizer, prompts, generations)
    print(f"Eval Correct: {all_correct} ({100*all_correct/total_val}%)")
    print(f"Eval Format Correct: {format_correct} ({100*format_correct/total_val}%)")
    print(f"Eval No Hit: {no_hit} ({100*no_hit/total_val}%)")
    print(f"Eval Entropy: {entropy}")


def run(argv=None):
    config = dict(
        model_dir = "./data/a5-alignment/models/Qwen2.5-Math-1.5B",
        tokenizer_dir = "./data/a5-alignment/models/Qwen2.5-Math-1.5B",
        input_dir = "./data/a5-alignment/MATH/train.jsonl",
        output_dir = "./data/a5-alignment/expert",
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
    output_dir = config.output_dir
    eval_dir = config.eval_dir
    prompt_template_dir = config.prompt_template_dir
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
    
    cum_loss, cum_token = 0, 0
    for expert_iter in range(5):
        print(f"Iteration {expert_iter}")
        load_policy_into_vllm_instance(model, vllm_model)
        output_file = get_expert_training_data(vllm_model, model, tokenizer, input_dir, prompt_template_dir, output_dir, expert_iter)
        eval_expert(vllm_model, model, tokenizer, eval_dir, prompt_template_dir)
        for t, train_batch in enumerate(batch(get_jsonl_data_stream(output_file, num_epochs=1), batch_size)):
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
                cum_loss, cum_token = 0, 0
                opt.step()
                opt.zero_grad()
            
    print("Final Iteration")
    eval_expert(vllm_model, model, tokenizer, eval_dir, prompt_template_dir)

    
if __name__ == "__main__":
    run()