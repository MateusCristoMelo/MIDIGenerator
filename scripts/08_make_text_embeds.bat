python ../src/08_make_text_embeds.py ^
  --csv ../data/splits/train_20k.csv ^
  --out_parquet ../data/text_embeds/train_20k.parquet

python ../src/08_make_text_embeds.py ^
  --csv ../data/splits/val_20k.csv ^
  --out_parquet ../data/text_embeds/val_20k.parquet
