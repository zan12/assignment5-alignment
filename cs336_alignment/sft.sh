for ne in 1 2 4 8 16; do
  for file in \
    "./data/a5-alignment/MATH/r1_distilled_train_stopped_shuf_128.jsonl" \
    "./data/a5-alignment/MATH/r1_distilled_train_stopped_shuf_256.jsonl" \
    "./data/a5-alignment/MATH/r1_distilled_train_stopped_shuf_512.jsonl" \
    "./data/a5-alignment/MATH/r1_distilled_train_stopped_shuf.jsonl" \
    "./data/a5-alignment/MATH/r1_distilled_train_stopped_correct.jsonl"; do
      echo "Running with num_epochs=$ne on $file"
      uv run -m cs336_alignment.sft \
        --num_epochs $ne \
        --input_dir "$file"
  done
done