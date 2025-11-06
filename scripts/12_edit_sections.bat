python ../src/12_edit_sections.py ^
  --bin_dir ../data/binpack ^
  --soft_ckpt ../runs/softprompt_midicaps/softprompt_ep02.pt ^
  --ctrl_vocab ../data/ctrl_vocab.json ^
  --out_dir ../data/edits ^
  --out_name gen_text_EDIT_8_12.mid ^
  --prompt_text "dark epic orchestral with choir and low strings" ^
  --source_midi ../data/infer_out/gen_text.mid ^
  --start_seconds 15 ^
  --end_seconds 20 ^
  --context_bars_before 6 ^
  --context_bars_after 2 ^
  --tempo_bpm 120 ^
  --time_signature 4/4 ^
  --temperature 1.2 ^
  --top_p 0.95 ^
  --max_new_tokens 3000 ^
  --allow_programs 40,41,42,43,44,45,46,47,48 ^
  --ban_drums

