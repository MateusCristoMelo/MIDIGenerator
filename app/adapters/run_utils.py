# adapters/run_utils.py
from __future__ import annotations
from pathlib import Path
from typing import Tuple
import re, time, shutil

RUN_PREFIX = "run_"

def start_new_run(base_out_dir: Path) -> Path:
    base_out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = base_out_dir / f"{RUN_PREFIX}{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def list_runs(base_out_dir: Path) -> list[Path]:
    if not base_out_dir.exists(): return []
    return sorted([p for p in base_out_dir.iterdir() if p.is_dir() and p.name.startswith(RUN_PREFIX)])

# ✔️ novo: aceita qualquer nome que COMEÇA com 4 dígitos e termina em .mid/.wav
_seq_re = re.compile(r"^(\d{4}).*\.(mid|wav)$", re.IGNORECASE)

def next_seq_id(run_dir: Path) -> str:
    run_dir.mkdir(parents=True, exist_ok=True)
    max_id = 0
    for p in run_dir.iterdir():
        m = _seq_re.match(p.name)
        if m:
            try:
                n = int(m.group(1))
                if n > max_id:
                    max_id = n
            except:
                pass
    return f"{max_id+1:04d}"

def seq_paths(run_dir: Path, seq_id: str) -> Tuple[Path, Path]:
    midi_path = run_dir / f"{seq_id}.mid"
    wav_path  = run_dir / f"{seq_id}.wav"
    return midi_path, wav_path

def mark_old(pair_path: Path) -> Path:
    """Renomeia 'xxxx.ext' -> 'xxxx-old.ext' (não sobrescreve)."""
    if not pair_path.exists(): return pair_path
    old = pair_path.with_name(pair_path.stem + "-old" + pair_path.suffix)
    i = 1
    while old.exists():
        old = pair_path.with_name(pair_path.stem + f"-old{i}" + pair_path.suffix)
        i += 1
    shutil.move(str(pair_path), str(old))
    return old
