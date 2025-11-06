python ../src/04_tokenize_midis.py ^
  --split_csv ../data/splits/train_20k.csv ^
  --ctrl_vocab ../data/ctrl_vocab_20k.json ^
  --out_jsonl ../data/tokens/train_20k.jsonl

python ../src/04_tokenize_midis.py ^
  --split_csv ../data/splits/val_20k.csv ^
  --ctrl_vocab ../data/ctrl_vocab_20k.json ^
  --out_jsonl ../data/tokens/val_20k.jsonl

python ../src/04_tokenize_midis.py ^
  --split_csv ../data/splits/test_20k.csv ^
  --ctrl_vocab ../data/ctrl_vocab_20k.json ^
  --out_jsonl ../data/tokens/test_20k.jsonl
