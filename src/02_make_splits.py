# 02_make_splits.py  (estratificado + sufixo automático)
import argparse, math
from pathlib import Path
import pandas as pd
import numpy as np

def print_split_stats(df_split, name, tempo_bucket, duration_bucket):
    g = df_split.copy()
    # recupera os buckets usados para estes índices
    g["tempo_bucket"] = tempo_bucket.loc[g.index].astype(str).values
    g["duration_bucket"] = duration_bucket.loc[g.index].astype(str).values
    tbl = (
        g.groupby(["tempo_bucket", "duration_bucket"], dropna=False)
         .size()
         .reset_index(name="count")
         .sort_values("count", ascending=False)
    )
    print(f"{name}: {len(g)} linhas")
    # mostra só os 10 maiores estratos, como no seu exemplo
    print(tbl.head(10).to_string(index=False))

def derive_out_paths(in_csv: Path):
    stem = in_csv.stem  # ex.: 'split_25k'
    suffix = "_" + stem.split("_", 1)[1] if "_" in stem else ""
    return in_csv.parent, suffix

def bin_tempo(tempo: pd.Series) -> pd.Series:
    bins = [-np.inf, 80, 120, 160, np.inf]
    labels = ["slow", "moderate", "fast", "very_fast"]
    return pd.cut(tempo.fillna(0), bins=bins, labels=labels, right=False).astype(str)

def bin_duration(dur: pd.Series) -> pd.Series:
    bins = [-np.inf, 15, 45, 120, 300, np.inf]
    labels = ["vshort", "short", "medium", "long", "vlong"]
    return pd.cut(dur.fillna(0), bins=bins, labels=labels, right=False).astype(str)

def stratified_fraction(df, by_col, frac, seed):
    if frac <= 0 or len(df) == 0:
        return df.iloc[0:0].copy()
    parts = []
    for k, g in df.groupby(by_col, dropna=False):
        n = int(round(len(g) * frac))
        n = max(0, min(n, len(g)))
        if n > 0:
            parts.append(g.sample(n=n, random_state=seed))
    return pd.concat(parts, axis=0) if parts else df.iloc[0:0].copy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True, type=str,
                    help="CSV de entrada (ex.: data/splits/split_25k.csv)")
    ap.add_argument("--train_frac", type=float, default=0.80)
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--test_frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    total = args.train_frac + args.val_frac + args.test_frac
    if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"As frações devem somar 1.0 (atual={total}).")

    in_csv = Path(args.in_csv)
    if not in_csv.exists():
        raise FileNotFoundError(in_csv)

    df = pd.read_csv(in_csv)
    if "midi_path" not in df.columns:
        raise KeyError("CSV precisa conter 'midi_path'.")

    # Buckets para estratificação
    if "tempo_word" in df.columns and df["tempo_word"].notna().any():
        tempo_bucket = df["tempo_word"].astype(str)
    else:
        if "tempo" not in df.columns:
            raise KeyError("Faltou 'tempo' ou 'tempo_word' para estratificar.")
        tempo_bucket = bin_tempo(df["tempo"])

    if "duration_word" in df.columns and df["duration_word"].notna().any():
        duration_bucket = df["duration_word"].astype(str)
    else:
        if "duration" not in df.columns:
            raise KeyError("Faltou 'duration' ou 'duration_word' para estratificar.")
        duration_bucket = bin_duration(df["duration"])

    df["_stratum"] = tempo_bucket + "|" + duration_bucket

    # Caso haja test_set=True, usa como teste e reparte o resto em train/val
    has_test_flag = "test_set" in df.columns and df["test_set"].astype(str).str.lower().isin(["true","1"]).any()
    if has_test_flag:
        df_test = df[df["test_set"].astype(str).str.lower().isin(["true","1"])].copy()
        df_rest = df[~df.index.isin(df_test.index)].copy()

        rem_total = args.train_frac + args.val_frac
        if rem_total <= 0:
            raise ValueError("Com test_set presente, train_frac + val_frac deve ser > 0.")
        train_rel = args.train_frac / rem_total
        # amostra estratificada para train dentro do restante
        df_train = stratified_fraction(df_rest, "_stratum", train_rel, args.seed)
        df_val = df_rest.loc[~df_rest.index.isin(df_train.index)].copy()
    else:
        # 1) escolhe teste estratificado
        df_test = stratified_fraction(df, "_stratum", args.test_frac, args.seed)
        df_rem = df.loc[~df.index.isin(df_test.index)].copy()
        # 2) escolhe val estratificado do restante
        rem_total = 1.0 - args.test_frac
        val_rel = args.val_frac / rem_total if rem_total > 0 else 0.0
        df_val = stratified_fraction(df_rem, "_stratum", val_rel, args.seed)
        # 3) resto é train
        df_train = df_rem.loc[~df_rem.index.isin(df_val.index)].copy()

    # Embaralha para não ficar em blocos
    df_train = df_train.sample(frac=1.0, random_state=args.seed).drop(columns=["_stratum"])
    df_val   = df_val.sample(frac=1.0, random_state=args.seed).drop(columns=["_stratum"])
    df_test  = df_test.sample(frac=1.0, random_state=args.seed).drop(columns=["_stratum"])

    out_dir, suffix = derive_out_paths(in_csv)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"train{suffix}.csv").write_text(df_train.to_csv(index=False), encoding="utf-8")
    (out_dir / f"val{suffix}.csv").write_text(df_val.to_csv(index=False), encoding="utf-8")
    (out_dir / f"test{suffix}.csv").write_text(df_test.to_csv(index=False), encoding="utf-8")

    print(f"Total: {len(df)} | train: {len(df_train)} | val: {len(df_val)} | test: {len(df_test)}")
    print(f"Salvos em: {out_dir.as_posix()}")
    print_split_stats(df_train, "train", tempo_bucket, duration_bucket)
    print_split_stats(df_val,   "val",   tempo_bucket, duration_bucket)
    print_split_stats(df_test,  "test",  tempo_bucket, duration_bucket)

if __name__ == "__main__":
    main()
