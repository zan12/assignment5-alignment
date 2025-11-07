import json
import re

from .drgrpo_grader import r1_zero_reward_fn


def postprocess(text):
    # find the last appearance of </think> ... <answer>
    # reformat as ... </think> <answer>
    match = list(re.finditer("</think>.*?<answer>", text, flags=re.DOTALL))
    if match:
        s, e = match[-1].start(), match[-1].end()
        text = text[:s] + text[s+8:e-8] + "</think> <answer>" + text[e:]
        return text
    
def process_file(input_file, output_file):
    with open(output_file, "w") as fo:
        with open(input_file, "r") as f:
            for line in f:
                jline = json.loads(line)
                jline["response"] = postprocess(jline["response"])
                if jline["response"] and jline["response"].strip().endswith("</answer>"):
                    reward = r1_zero_reward_fn(jline["response"], jline["solution"])
                    if reward["reward"] == 1.0:
                        json.dump(jline, fo)
                        fo.write("\n")

if __name__ == "__main__":
    input_file = "./data/a5-alignment/MATH/r1_distilled_train.jsonl"
    output_file = "./data/a5-alignment/MATH/r1_distilled_train_correct.jsonl"
    process_file(input_file, output_file)
    # text = "</think>\nhello<answer> aonan"
    # print(postprocess(text))