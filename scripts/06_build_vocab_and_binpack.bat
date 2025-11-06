python ../src/06_build_vocab_and_binpack.py ^
  --train_jsonl ../data/tokens/train_20k.jsonl ^
  --val_jsonl   ../data/tokens/val_20k.jsonl ^
  --test_jsonl  ../data/tokens/test_20k.jsonl ^
  --ctrl_vocab  ../data/ctrl_vocab_20k.json ^
  --out_dir     ../data/binpack_20k ^
  --seq_len     1024 ^
  --shard_tokens 1000000
