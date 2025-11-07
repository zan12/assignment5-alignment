import argparse
import json

from collections import Counter
from typing import Any, Callable, Iterator, List
from vllm import LLM, SamplingParams

from .drgrpo_grader import r1_zero_reward_fn


def make_config(config, argv):
    parser = argparse.ArgumentParser()
    for k, v in config.items():
        parser.add_argument(f"--{k}", dest=k, default=v)
    config = parser.parse_args(argv)
    return config

def apply_prompt_template(text: str, prompt_template_dir: str):
    with open(prompt_template_dir, "r") as f:
        prompt_template = f.read()
    return prompt_template.format(question=text)

def get_jsonl_data_stream(data_dir: str, num_epochs: int = 1) -> Iterator[dict[str, Any]]:
    for _ in range(num_epochs):
        with open(data_dir, "r") as f:
            for line in f:
                yield json.loads(line)

def make_math_eval_data(data_dir, prompt_template_dir):
    prompts, ground_truths = [], []
    for jline in get_jsonl_data_stream(data_dir):
        prompts.append(
            apply_prompt_template(jline["problem"], prompt_template_dir)
        )
        ground_truths.append(jline["solution"])
    return prompts, ground_truths

def evaluate_vllm(
    vllm_model: LLM,
    prompts: List[str],
    ground_truths: List[str],
    reward_fn: Callable[[str, str], dict[str, float]] = r1_zero_reward_fn,
    eval_sampling_params: SamplingParams = SamplingParams(temperature=1., top_p=1., min_tokens=4, max_tokens=1024, stop=["</answer>"], include_stop_str_in_output=True,),
) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """

    outputs = vllm_model.generate(prompts, eval_sampling_params)
    all_summarize = []
    for output, gt in zip(outputs, ground_truths):
        prompt = output.prompt
        for o in output.outputs:
            generated_text = o.text
            rewards = reward_fn(generated_text, gt)
            summarize = {
                "prompt": prompt,
                "gt": gt,
                "generation": generated_text,
                "rewards": rewards,
            }
            all_summarize.append(summarize)
    return all_summarize


def run(argv=None):
    config = dict(
        eval_data_dir = "./data/a5-alignment/MATH/validation.jsonl",
        prompt_template_dir = "./cs336_alignment/prompts/r1_zero.prompt",
        model_dir="./data/a5-alignment/models/Qwen2.5-Math-1.5B",
        output_dir="/tmp/eval_result.jsonl"
    )
    config = make_config(config, argv)
    eval_data_dir = config.eval_data_dir
    prompt_template_dir = config.prompt_template_dir
    model_dir = config.model_dir
    output_dir = config.output_dir
    prompts, ground_truths = make_math_eval_data(
        eval_data_dir, prompt_template_dir
    )
    vllm_model = LLM(model=model_dir)
    rewards = evaluate_vllm(
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
    print(Counter(map(str, rewards)))

if __name__ == "__main__":
    run()