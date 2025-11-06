python ../src/03_make_ctrl_vocab.py ^
  --in_csv ../data/splits/train_20k.csv ^
  --out_json ../data/ctrl_vocab_20k.json ^
  --min_bpm 60 --max_bpm 180 --bpm_step 2 ^
  --bars_tokens "4,8,12,16,24,32,48,64"