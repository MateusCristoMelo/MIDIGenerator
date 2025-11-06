import argparse, json, math
from pathlib import Path
from collections import OrderedDict

import numpy as np

SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3

def load_ctrl_tokens(ctrl_vocab_path: Path) -> list[str]:
    with open(ctrl_vocab_path, "r", encoding="utf-8") as f:
        v = json.load(f)
    out = []
    # tempo
    out += list(v.get("tempo_bpm", {}).get("tokens", []) or [])
    # length bars
    out += list(v.get("length_bars", {}).get("tokens", []) or [])
    # time signature (opcional)
    out += list(v.get("time_signature", {}).get("tokens", []) or [])
    # key (opcional)
    out += list(v.get("key", {}).get("tokens", []) or [])
    # dedup preservando ordem
    seen = set(); dedup = []
    for t in out:
        if t not in seen:
            seen.add(t); dedup.append(t)
    return dedup

def iter_jsonl_tokens(jsonl_path: Path):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            toks = rec.get("tokens", [])
            # flatten defensivo (evita [[...]])
            if len(toks) and isinstance(toks[0], (list, tuple)):
                toks = toks[0]
            yield toks

def build_vocab(train_jsonl: Path, ctrl_vocab: Path, include_extra_for_vocab: list[Path] | None = None) -> dict[str, int]:
    # ordem: especiais -> CTRL -> tokens observados
    vocab = OrderedDict()
    for t in SPECIAL_TOKENS:
        vocab[t] = len(vocab)

    # CTRL tokens (range completo que definimos)
    for t in load_ctrl_tokens(ctrl_vocab):
        if t not in vocab:
            vocab[t] = len(vocab)

    # tokens observados (train sempre; extra = val/test opcional)
    def add_from(jsonl_path: Path):
        for toks in iter_jsonl_tokens(jsonl_path):
            for t in toks:
                ts = t if isinstance(t, str) else str(t)
                if ts.startswith("<CTRL_>"):  # proteção inútil, mas segura contra algo malformado
                    continue
                if ts not in vocab:
                    vocab[ts] = len(vocab)

    add_from(train_jsonl)
    for p in (include_extra_for_vocab or []):
        if p and p.exists():
            add_from(p)

    return dict(vocab)

def encode_sequence(tokens: list, stoi: dict[str, int]) -> list[int]:
    ids = []
    # BOS
    ids.append(BOS_ID)
    for t in tokens:
        ts = t if isinstance(t, str) else str(t)
        i = stoi.get(ts, UNK_ID)
        ids.append(i)
    # EOS
    ids.append(EOS_ID)
    return ids

def binpack_split(jsonl_path: Path, stoi: dict[str, int], out_dir: Path, split_name: str,
                  seq_len: int, shard_tokens: int, drop_remainder: bool = True):
    """
    Concatena todas as sequências do split e fatia em blocos de seq_len.
    Grava shards .npy com dtype=uint32.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    sequences_per_shard = max(1, shard_tokens // seq_len)

    buffer = []
    n_tokens_total = 0
    shard_idx = 0
    sequences_in_current_shard = []

    def flush_shard():
        nonlocal sequences_in_current_shard, shard_idx
        if not sequences_in_current_shard:
            return
        arr = np.stack(sequences_in_current_shard, axis=0).astype(np.uint32)  # [N, seq_len]
        shard_path = out_dir / f"{split_name}_len{seq_len}_shard{shard_idx:04d}.npy"
        np.save(shard_path, arr, allow_pickle=False)
        shard_idx += 1
        sequences_in_current_shard = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            toks = rec.get("tokens", [])
            if len(toks) and isinstance(toks[0], (list, tuple)):
                toks = toks[0]
            ids = encode_sequence(toks, stoi)

            buffer.extend(ids)
            n_tokens_total += len(ids)

            # corta em pedaços de seq_len
            while len(buffer) >= seq_len:
                seq = buffer[:seq_len]
                buffer = buffer[seq_len:]
                sequences_in_current_shard.append(np.array(seq, dtype=np.uint32))
                if len(sequences_in_current_shard) >= sequences_per_shard:
                    flush_shard()

    # trata o resto:
    if len(buffer) > 0 and not drop_remainder:
        # pad até seq_len
        pad_needed = seq_len - len(buffer)
        if pad_needed > 0:
            buffer.extend([PAD_ID] * pad_needed)
        sequences_in_current_shard.append(np.array(buffer[:seq_len], dtype=np.uint32))
    flush_shard()

    # grava um index/meta simples
    meta = {
        "split": split_name,
        "seq_len": seq_len,
        "n_tokens_total": int(n_tokens_total),
        "n_shards": int(shard_idx),
        "shard_tokens_target": int(shard_tokens),
        "special_ids": {"PAD": PAD_ID, "BOS": BOS_ID, "EOS": EOS_ID, "UNK": UNK_ID},
    }
    with open(out_dir / f"{split_name}_index.json", "w", encoding="utf-8") as fmeta:
        json.dump(meta, fmeta, ensure_ascii=False, indent=2)

    print(f"[{split_name}] tokens={n_tokens_total} | shards={shard_idx}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", required=True, help="ex.: data/tokens/train_5k.jsonl")
    ap.add_argument("--val_jsonl",   required=True, help="ex.: data/tokens/val_5k.jsonl")
    ap.add_argument("--test_jsonl",  required=True, help="ex.: data/tokens/test_5k.jsonl")
    ap.add_argument("--ctrl_vocab",  required=True, help="ex.: data/ctrl_vocab.json")
    ap.add_argument("--out_dir",     default="data/binpack", help="pasta de saída")
    ap.add_argument("--seq_len",     type=int, default=1024)
    ap.add_argument("--shard_tokens",type=int, default=1_000_000,
                    help="aprox de tokens por shard (seq_len * nseq por shard)")
    ap.add_argument("--include_val_in_vocab", action="store_true",
                    help="inclui val no vocabulário (além de train)")
    ap.add_argument("--include_test_in_vocab", action="store_true",
                    help="inclui test no vocabulário (além de train)")
    ap.add_argument("--keep_last_partial", action="store_true",
                    help="mantém o último pedaço parcial com padding em vez de descartar")
    args = ap.parse_args()

    train_p = Path(args.train_jsonl)
    val_p   = Path(args.val_jsonl)
    test_p  = Path(args.test_jsonl)
    ctrl_p  = Path(args.ctrl_vocab)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- VOCAB ---
    extras = []
    if args.include_val_in_vocab:
        extras.append(val_p)
    if args.include_test_in_vocab:
        extras.append(test_p)
    stoi = build_vocab(train_p, ctrl_p, include_extra_for_vocab=extras)

    # salva vocab
    with open(out_dir / "vocab.json", "w", encoding="utf-8") as fv:
        json.dump(stoi, fv, ensure_ascii=False, indent=2)
    # e inverso
    itos = {int(v): k for k, v in stoi.items()}
    with open(out_dir / "vocab_itos.json", "w", encoding="utf-8") as fi:
        json.dump(itos, fi, ensure_ascii=False, indent=2)

    print(f"Vocab salvo em {out_dir/'vocab.json'} | tamanho={len(stoi)}")
    print("IDs especiais:", {"PAD": PAD_ID, "BOS": BOS_ID, "EOS": EOS_ID, "UNK": UNK_ID})

    # --- BINPACK por split ---
    binpack_split(train_p, stoi, out_dir, "train", args.seq_len, args.shard_tokens, drop_remainder=not args.keep_last_partial)
    binpack_split(val_p,   stoi, out_dir, "val",   args.seq_len, args.shard_tokens, drop_remainder=not args.keep_last_partial)
    binpack_split(test_p,  stoi, out_dir, "test",  args.seq_len, args.shard_tokens, drop_remainder=not args.keep_last_partial)

if __name__ == "__main__":
    main()
