# 05_roundtrip_smoke.py  (Miditok v3 + symusic friendly)
import argparse, json
from pathlib import Path
from miditok import TokenizerConfig, REMI

# tenta usar o helper do Miditok para salvar symusic.MIDI
try:
    from miditok.utils import save_midi as mt_save_midi
except Exception:
    mt_save_midi = None

def build_tokenizer():
    cfg = TokenizerConfig(
        beat_res={(0, 4): 8},   # mesma resolução do 04
        num_velocities=32,
        use_chords=False,
        use_rests=True,
        use_tempos=True,
        use_time_signatures=True,
        use_programs=True,
        one_token_stream_for_programs=True,
    )
    return REMI(cfg)

def save_symusic_midi(midi_obj, out_path: Path):
    """
    Tenta salvar o objeto MIDI retornado pelo Miditok (symusic) de forma robusta.
    Ordem de tentativas:
    1) miditok.utils.save_midi (se disponível)
    2) métodos comuns no objeto: dump_midi, dump, save_midi, save, write, write_midi
    3) fallback para bytes: to_bytes()
    4) tenta atributos internos como .midi com .save/.dump/.write
    """
    # 1) helper oficial (se existir na sua versão do Miditok)
    if mt_save_midi is not None:
        try:
            mt_save_midi(midi_obj, out_path)
            return
        except Exception:
            pass

    # 2) métodos comuns
    for m in ("dump_midi", "dump", "save_midi", "save", "write", "write_midi"):
        if hasattr(midi_obj, m):
            meth = getattr(midi_obj, m)
            try:
                meth(out_path)            # Path
                return
            except TypeError:
                try:
                    meth(str(out_path))   # str
                    return
                except Exception:
                    pass
            except Exception:
                pass

    # 3) fallback para bytes
    for m in ("to_bytes",):
        if hasattr(midi_obj, m):
            try:
                data = getattr(midi_obj, m)()
                with open(out_path, "wb") as f:
                    f.write(data)
                return
            except Exception:
                pass

    # 4) tenta objeto interno
    for inner in ("midi", "mid", "mido_obj"):
        if hasattr(midi_obj, inner):
            inner_obj = getattr(midi_obj, inner)
            for m in ("save", "dump", "write"):
                if hasattr(inner_obj, m):
                    try:
                        getattr(inner_obj, m)(str(out_path))
                        return
                    except Exception:
                        pass

    raise RuntimeError("Não sei salvar este objeto MIDI (symusic).")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True, help="ex.: data/tokens/train_5k.jsonl")
    ap.add_argument("--out_dir", required=True, help="ex.: data/roundtrip/")
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = build_tokenizer()
    written = 0

    with open(args.in_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if written >= args.n:
                break
            rec = json.loads(line)
            tokens = rec["tokens"]

            # remove CTRL tokens do início
            i0 = 0
            while i0 < len(tokens) and isinstance(tokens[i0], str) and tokens[i0].startswith("<CTRL_"):
                i0 += 1
            musical = tokens[i0:]

            # garante 1D
            if len(musical) > 0 and isinstance(musical[0], (list, tuple)):
                musical = musical[0]

            try:
                midi_obj = tok.tokens_to_midi(musical)  # Miditok v3 espera 1D
                out_path = out_dir / f"roundtrip_{written+1}.mid"
                save_symusic_midi(midi_obj, out_path)
                written += 1
            except Exception as e:
                with open(out_dir / "roundtrip_fail.log", "a", encoding="utf-8") as ferr:
                    ferr.write(f"{rec.get('midi_path')} :: {repr(e)}\n")

    print(f"Gerados {written} MIDIs em {out_dir}")

if __name__ == "__main__":
    main()