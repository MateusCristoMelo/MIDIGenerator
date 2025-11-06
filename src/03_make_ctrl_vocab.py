import argparse, json, math
from pathlib import Path
import pandas as pd

GM_FAMILY_BY_RANGE = [
    (0,   7,  "PIANO"),
    (8,  15,  "CHROMATIC_PERC"),
    (16, 23,  "ORGAN"),
    (24, 31,  "GUITAR"),
    (32, 39,  "BASS"),
    (40, 47,  "STRINGS"),
    (48, 55,  "ENSEMBLE"),
    (56, 63,  "BRASS"),
    (64, 71,  "REED"),
    (72, 79,  "PIPE"),
    (80, 87,  "SYNTH_LEAD"),
    (88, 95,  "SYNTH_PAD"),
    (96,103,  "SYNTH_EFFECTS"),
    (104,111, "ETHNIC"),
    (112,119, "PERCUSSIVE"),
    (120,127, "SFX"),
]

def gm_family(program: int) -> str:
    for lo, hi, fam in GM_FAMILY_BY_RANGE:
        if lo <= program <= hi:
            return fam
    return "UNKNOWN"

def parse_timesig(ts: str):
    # espera "4/4", "3/4", "6/8"...
    try:
        num, den = ts.split("/")
        num, den = int(num), int(den)
        if num <= 0 or den <= 0:
            return None
        return num, den
    except Exception:
        return None

def bars_from_seconds(duration_s: float, bpm: float, ts: str) -> float:
    # quarter-notes por compasso = num * (4/den)
    parsed = parse_timesig(ts)
    if not parsed or not bpm or bpm <= 0 or duration_s <= 0:
        return None
    num, den = parsed
    qnotes_per_bar = num * (4.0 / den)
    # barras = (duração * bpm) / (60 * qnotes_por_barra)
    return (duration_s * bpm) / (60.0 * qnotes_per_bar)

def quantize_bars(bars: float, allowed: list[int]) -> int | None:
    if bars is None or bars <= 0:
        return None
    # escolhe o mais próximo
    return min(allowed, key=lambda x: abs(x - bars))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True, help="ex.: data/splits/train_25k.csv")
    ap.add_argument("--out_json", default="data/ctrl_vocab.json")
    ap.add_argument("--min_bpm", type=int, default=60)
    ap.add_argument("--max_bpm", type=int, default=180)
    ap.add_argument("--bpm_step", type=int, default=2)
    ap.add_argument("--bars_tokens", type=str, default="4,8,12,16,24,32,48,64",
                    help="lista de comprimentos em barras para tokenizar (separados por vírgula)")
    ap.add_argument("--top_keys", type=int, default=12)
    ap.add_argument("--top_ts", type=int, default=3)
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    if "midi_path" not in df.columns:
        raise KeyError("CSV precisa ter 'midi_path'.")

    # --- TEMPO TOKENS ---
    min_bpm = args.min_bpm
    max_bpm = args.max_bpm
    step = max(1, args.bpm_step)
    tempo_values = list(range(min_bpm, max_bpm + 1, step))
    tempo_tokens = [f"<CTRL_TEMPO_{b}>" for b in tempo_values]

    # --- BARS TOKENS (a partir de segundos do dataset) ---
    allowed_bars = [int(x.strip()) for x in args.bars_tokens.split(",") if x.strip().isdigit()]
    # tenta calcular barras observadas para estatística (opcional)
    bars_obs = []
    if all(c in df.columns for c in ["duration", "tempo", "time_signature"]):
        for d, bpm, ts in zip(df["duration"], df["tempo"], df["time_signature"]):
            bars = bars_from_seconds(d, bpm, str(ts))
            if bars:
                qb = quantize_bars(bars, allowed_bars)
                if qb:
                    bars_obs.append(qb)
    bars_tokens = [f"<CTRL_LEN_{b}BARS>" for b in allowed_bars]

    # --- TIME SIGNATURES / KEYS (opcionais) ---
    ts_list = []
    if "time_signature" in df.columns:
        ts_list = (
            df["time_signature"].dropna().astype(str).value_counts().head(args.top_ts).index.tolist()
        )
    key_list = []
    if "key" in df.columns:
        key_list = (
            df["key"].dropna().astype(str).value_counts().head(args.top_keys).index.tolist()
        )
    ts_tokens  = [f"<CTRL_TS_{ts.replace('/','_')}>" for ts in ts_list]
    key_tokens = [f"<CTRL_KEY_{k.replace(' ', '')}>" for k in key_list]

    # --- INSTRUMENTOS GM ---
    gm_programs = set()
    if "instrument_numbers_sorted" in df.columns:
        # coluna costuma vir como string de lista -> converter
        def parse_list(x):
            if pd.isna(x): return []
            s = str(x).strip()
            # tenta json-like
            if s.startswith("[") and s.endswith("]"):
                try:
                    import ast
                    li = ast.literal_eval(s)
                    return [int(v) for v in li if str(v).isdigit() or isinstance(v, int)]
                except Exception:
                    pass
            # fallback: separado por vírgula
            parts = [p.strip() for p in s.split(",")]
            out = []
            for p in parts:
                try: out.append(int(p))
                except: pass
            return out

        df["_gm_numbers"] = df["instrument_numbers_sorted"].apply(parse_list)
        for li in df["_gm_numbers"]:
            for prg in li:
                if 0 <= prg <= 127:
                    gm_programs.add(int(prg))
        df.drop(columns=["_gm_numbers"], inplace=True, errors="ignore")

    gm_programs = sorted(gm_programs)
    gm_families = {str(p): gm_family(p) for p in gm_programs}

    # --- MONTA O VOCAB ---
    vocab = {
        "tempo_bpm": {
            "min": min_bpm, "max": max_bpm, "step": step,
            "tokens": tempo_tokens
        },
        "length_bars": {
            "allowed": allowed_bars,
            "tokens": bars_tokens,
            "coverage_sampled": {str(b): int(b in bars_obs) for b in allowed_bars} if bars_obs else None
        },
        "time_signature": {
            "top": ts_list,
            "tokens": ts_tokens
        },
        "key": {
            "top": key_list,
            "tokens": key_tokens
        },
        "gm_programs_seen": gm_programs,
        "gm_family_map": gm_families,
        "notes": {
            "duration_seconds_input": "interface recebe segundos e converte para barras internamente",
            "ts_default_if_missing": "4/4",
            "key_optional": True,
            "ts_optional": True
        }
    }

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print("CTRL vocab salvo em:", out)
    print("- #tempo_tokens:", len(tempo_tokens))
    print("- length_bars tokens:", bars_tokens)
    print("- top time_signatures:", ts_list)
    print("- top keys:", key_list[:10])
    print("- gm_programs vistos:", len(gm_programs))
    fam_counts = {}
    for p in gm_programs:
        fam_counts[gm_families[str(p)]] = fam_counts.get(gm_families[str(p)], 0) + 1
    print("- gm famílias (contagem):", dict(sorted(fam_counts.items(), key=lambda x: -x[1])))
    
if __name__ == "__main__":
    main()
