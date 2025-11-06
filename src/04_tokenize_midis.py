# 04_tokenize_midis.py
import argparse, json
from pathlib import Path
import pandas as pd

from miditok import TokenizerConfig, REMI

# ---------- helpers de controle ----------
def parse_ts(ts: str):
    try:
        num, den = str(ts).split("/")
        num, den = int(num), int(den)
        assert num > 0 and den > 0
        return num, den
    except Exception:
        return 4, 4  # default

def quantize_bpm(bpm: float, min_bpm: int, max_bpm: int, step: int) -> int:
    if bpm is None or bpm <= 0:
        return 120  # default
    bpm = max(min_bpm, min(max_bpm, bpm))
    # aproxima ao grid
    base = round((bpm - min_bpm) / step) * step + min_bpm
    return max(min_bpm, min(max_bpm, base))

def bars_from_seconds(duration_s: float, bpm: float, ts: str) -> float:
    num, den = parse_ts(ts)
    qnotes_per_bar = num * (4.0 / den)
    return (duration_s * bpm) / (60.0 * qnotes_per_bar) if (duration_s and bpm) else None

def quantize_bars(bars: float, allowed: list[int]) -> int:
    if not bars or bars <= 0:
        return allowed[0]
    return min(allowed, key=lambda x: abs(x - bars))

def build_ctrl_tokens(row, vocab):
    # tempo
    tconf = vocab["tempo_bpm"]
    bpm = row["tempo"] if "tempo" in row and pd.notna(row["tempo"]) else None
    qbpm = quantize_bpm(bpm, tconf["min"], tconf["max"], tconf["step"])
    t_tok = f"<CTRL_TEMPO_{qbpm}>"

    # bars
    lconf = vocab["length_bars"]
    ts = row["time_signature"] if "time_signature" in row and pd.notna(row["time_signature"]) else "4/4"
    dur = row["duration"] if "duration" in row and pd.notna(row["duration"]) else None
    bars = bars_from_seconds(dur, qbpm, ts)
    qbars = quantize_bars(bars, lconf["allowed"])
    len_tok = f"<CTRL_LEN_{qbars}BARS>"

    # time signature (opcional; só se estiver entre os top)
    ts_tokens_whitelist = set(vocab.get("time_signature", {}).get("tokens", []))
    ts_tok = f"<CTRL_TS_{str(ts).replace('/','_')}>"
    ts_ctrl = [ts_tok] if ts_tok in ts_tokens_whitelist else []

    # key (opcional; idem)
    key_val = row["key"] if "key" in row and pd.notna(row["key"]) else None
    key_tokens_whitelist = set(vocab.get("key", {}).get("tokens", []))
    key_tok = f"<CTRL_KEY_{str(key_val).replace(' ', '')}>"
    key_ctrl = [key_tok] if key_val and key_tok in key_tokens_whitelist else []

    return [t_tok, len_tok] + ts_ctrl + key_ctrl

# ---------- tokenizador ----------
def build_tokenizer():
    # Config REMI simples, com stream único e programas habilitados
    cfg = TokenizerConfig(
        beat_res={(0, 4): 8},           # grade ~1/16 (4 posições por batida)
        num_velocities=32,
        use_chords=False,
        use_rests=True,
        use_tempos=True,
        use_time_signatures=True,
        use_programs=True,              # importante para multitrack
        one_token_stream_for_programs=True
    )
    return REMI(cfg)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_csv", required=True, help="ex.: data/splits/train_25k.csv")
    ap.add_argument("--ctrl_vocab", required=True, help="ex.: data/ctrl_vocab.json")
    ap.add_argument("--out_jsonl", required=True, help="ex.: data/tokens/train_25k.jsonl")
    ap.add_argument("--max_files", type=int, default=0, help="limite opcional para smoke run")
    args = ap.parse_args()

    df = pd.read_csv(args.split_csv)
    with open(args.ctrl_vocab, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tok = build_tokenizer()

    n = len(df) if args.max_files <= 0 else min(args.max_files, len(df))
    count_ok, count_fail = 0, 0

    with out_path.open("w", encoding="utf-8") as fout:
        for i, row in enumerate(df.itertuples(index=False), 1):
            if i > n:
                break
            midi_path = getattr(row, "midi_path")
            try:
                # ✅ Passe o caminho diretamente para o Miditok
                seqs = tok.midi_to_tokens(Path(midi_path))

                # Miditok pode retornar 1 TokSequence ou lista
                seq = seqs[0].tokens if isinstance(seqs, list) else seqs.tokens

                # monta controls
                r = df.iloc[i-1]
                ctrl = build_ctrl_tokens(r, vocab)

                record = {
                    "midi_path": midi_path,
                    "controls": ctrl,
                    "tokens": [*ctrl, *seq]
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                count_ok += 1
            except Exception as e:
                count_fail += 1
                with (out_path.parent / (out_path.stem + "_fail.log")).open("a", encoding="utf-8") as ferr:
                    ferr.write(f"{midi_path}\t{repr(e)}\n")

    print(f"Tokenização concluída: OK={count_ok} | FAIL={count_fail}")
    print("Saída:", out_path)

if __name__ == "__main__":
    main()