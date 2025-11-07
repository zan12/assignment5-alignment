import argparse
import json
import tqdm
from together import Together

from .eval_math import apply_prompt_template, get_jsonl_data_stream


def get_llm_completion_stream(client, model: str, prompt: str):
    return client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        top_p=0.9,
        max_tokens=1024,
        # stop=["</answer>"], # Will stop at eos.
        # include_stop_str_in_output=True, # Does not support adding stop str at the end.
    )


def run(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--math_train_data_dir",
        dest = "math_train_data_dir",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        dest = "output_dir",
        required=True,
    )
    config = parser.parse_args(argv)

    client = Together(
        api_key="5b8bf7971c89e9d4bd5f05ff88ca5a6da13d98698175a993ba1baf4f5c016f6f",
    )
    math_train_data_dir = config.math_train_data_dir  # "./data/a5-alignment/MATH/train_10_examples.jsonl"
    math_prompt_template_dir = "./cs336_alignment/prompts/r1_zero.prompt"
    output_dir = config.output_dir  # "./data/a5-alignment/MATH/r1_distill.jsonl"
    model = "deepseek-ai/DeepSeek-R1" # "deepseek-ai/DeepSeek-R1:fireworks-ai"

    with open(output_dir, "w") as fo:
        for jline in tqdm.tqdm(get_jsonl_data_stream(math_train_data_dir)):
            prompt = apply_prompt_template(jline["problem"], math_prompt_template_dir)
            completion = get_llm_completion_stream(client, model, prompt)
            jline["prompt"] = prompt
            jline["response"] = completion.choices[0].message.content
            json.dump(jline, fo)
            fo.write("\n")


if __name__ == "__main__":
    run()
    