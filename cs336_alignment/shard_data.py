import os

# input file
input_file = "./data/a5-alignment/MATH/train.jsonl"

# number of shards
num_shards = 10

# read all lines
with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

total = len(lines)
rows_per_shard = total // num_shards  # should be 750 if 7500 rows total

# create output folder
os.makedirs("shards", exist_ok=True)

for i in range(num_shards):
    start = i * rows_per_shard
    end = (i + 1) * rows_per_shard
    shard_lines = lines[start:end]

    shard_file = f"./data/a5-alignment/MATH/train_shard_{i}.jsonl"
    with open(shard_file, "w", encoding="utf-8") as f:
        f.writelines(shard_lines)

    print(f"Shard {i+1}: {len(shard_lines)} rows → {shard_file}")