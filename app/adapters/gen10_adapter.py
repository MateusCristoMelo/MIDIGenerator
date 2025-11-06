# adapters/gen10_adapter.py
# Adapter para o script 10_infer_text_softprompt.py (inferência)
# - Chamada simples a partir do app.py
# - Converte "barras -> segundos" com base em BPM e compasso
# - (Opcional) renderiza WAV via pretty_midi.fluidsynth se soundfont for fornecido

from __future__ import annotations
import subprocess, sys, time, io, shutil
from pathlib import Path
from typing import Optional, Dict, Any
from adapters.run_utils import next_seq_id, seq_paths
import pretty_midi, soundfile as sf

def _parse_ts(ts: str) -> tuple[int, int]:
    """Ex.: '4/4' -> (4, 4)."""
    try:
        n, d = ts.split("/")
        return int(n), int(d)
    except Exception:
        return 4, 4

def bars_to_seconds(bars: int, bpm: int, time_signature: str) -> float:
    """Converte #barras -> segundos, respeitando n/d do compasso."""
    n, d = _parse_ts(time_signature)
    quarter_per_bar = n * (4.0 / d)  # quantas semínimas existem por barra
    seconds_per_bar = (quarter_per_bar * 60.0) / max(1, bpm)
    return bars * seconds_per_bar

def _render_wav(midi_path: Path, wav_path: Path, soundfont: Path) -> bytes | None:
    """
    Tenta renderizar WAV (44.1 kHz) a partir de um MIDI usando:
    1) pretty_midi.PrettyMIDI().fluidsynth(fs=44100, sf2_path=...)
    2) fallback: CLI 'fluidsynth -ni <sf2> <mid> -F <wav> -r 44100'
    Retorna os bytes do WAV se der certo; senão, None.
    """
    # Caminho 1: pretty_midi
    try:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
        audio = pm.fluidsynth(fs=44100, sf2_path=str(soundfont))  # <-- 'fs' é o correto
        if audio is not None and audio.size > 0:
            # normaliza leve para evitar clip
            mx = float(audio.max()) if audio.size else 0.0
            if mx > 1.0:
                audio = audio / (mx + 1e-9)
            wav_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(wav_path), audio, 44100, subtype="PCM_16")
            return wav_path.read_bytes()
    except Exception as e:
        print(f"[gen10_adapter] pretty_midi render falhou: {e}")

    # Caminho 2: fluidsynth CLI (se instalado)
    try:
        if shutil.which("fluidsynth"):
            cmd = [
                "fluidsynth", "-ni",
                str(soundfont),
                str(midi_path),
                "-F", str(wav_path),
                "-r", "44100",
            ]
            subprocess.run(cmd, check=True)
            if wav_path.exists():
                return wav_path.read_bytes()
    except Exception as e:
        print(f"[gen10_adapter] fluidsynth CLI falhou: {e}")

    return None


def generate_v10(
    *,
    # caminhos do seu projeto
    repo_root: Path,              # raiz do projeto (ex.: Path.cwd())
    bin_dir: Path,                # ex.: repo_root / "data/binpack"
    soft_ckpt: Path,              # ex.: repo_root / "runs/softprompt_midicaps/softprompt_ep02.pt"
    ctrl_vocab: Path,             # ex.: repo_root / "data/ctrl_vocab.json"
    out_dir: Path,                # ex.: repo_root / "data/infer_out"
    script10: Optional[Path] = None,  # caminho do 10_infer_text_softprompt.py; default resolve abaixo
    # parâmetros criativos
    prompt_text: str,
    bpm: int = 120,
    time_signature: str = "4/4",
    duration_seconds: int = 30,
    temperature: float = 1.0,
    top_p: float = 0.9,
    allow_programs: str = "",     # "40,41,42,..." (GM)
    ban_drums: bool = False,
    force_program: int = -1,      # se >=0, injeta Program_N após os CTRLs
    # primer opcional
    primer_bars: int = 0,
    primer_tokens: int = 0,
    # retrieval/primer guiado (opcional, se você usa midicaps)
    captions_csv: Optional[Path] = None,
    tokens_jsonl: Optional[Path] = None,
    # renderização de áudio (opcional)
    soundfont: Optional[Path] = None,
    # limites
    max_new_tokens: int = 5000,
    run_dir: Path,
    seq_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Executa o script 10 e retorna caminhos e bytes gerados.

    Retorno:
      {
        "ok": bool,
        "midi_path": Path|None,
        "wav_path": Path|None,
        "wav_bytes": bytes|None,
        "stdout": str,
        "stderr": str,
        "cmd": list[str],
        "meta": {...}
      }
    """
    script10 = script10 or (repo_root / "src" / "10_infer_text_softprompt.py")
    assert script10.exists(), f"script não encontrado: {script10}"
    assert bin_dir.exists(), f"bin_dir não encontrado: {bin_dir}"
    assert soft_ckpt.exists(), f"soft_ckpt não encontrado: {soft_ckpt}"
    assert ctrl_vocab.exists(), f"ctrl_vocab não encontrado: {ctrl_vocab}"
    out_dir.mkdir(parents=True, exist_ok=True)
 

    # timestamp antes de rodar o script
    t0 = time.time()

    # nome de saída
    run_dir.mkdir(parents=True, exist_ok=True)
    seq_id = seq_id or next_seq_id(run_dir)
    midi_path, wav_path = seq_paths(run_dir, seq_id)

    # monta comando (mantendo compat. com script 10)
    cmd = [
        sys.executable, str(script10),
        "--bin_dir", str(bin_dir),
        "--soft_ckpt", str(soft_ckpt),
        "--ctrl_vocab", str(ctrl_vocab),
        "--out_dir", str(out_dir),
        "--prompt_text", str(prompt_text),
        "--tempo_bpm", str(bpm),
        "--time_signature", str(time_signature),
        "--duration_seconds", f"{duration_seconds:.6f}",
        "--max_new_tokens", str(max_new_tokens),
        "--temperature", str(temperature),
        "--top_p", str(top_p),
    ]

    if allow_programs:
        cmd += ["--allow_programs", allow_programs]
    if ban_drums:
        cmd += ["--ban_drums"]
    if force_program is not None and int(force_program) >= 0:
        cmd += ["--force_program", str(int(force_program))]
    if primer_bars and int(primer_bars) > 0:
        cmd += ["--primer_bars", str(int(primer_bars))]
    if primer_tokens and int(primer_tokens) > 0:
        cmd += ["--primer_tokens", str(int(primer_tokens))]
    if captions_csv:
        cmd += ["--captions_csv", str(captions_csv)]
    if tokens_jsonl:
        cmd += ["--tokens_jsonl", str(tokens_jsonl)]

    # Executa
    proc = subprocess.run(
        cmd,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    ok = (proc.returncode == 0)

    # O script 10 salva no --out_dir; se você já controla o nome do arquivo lá dentro,
    # pode localizar pelo padrão. Aqui assumimos que você renomeia/organiza depois.
    # Como fallback, procure o .mid mais novo.
    # Descobrir o .mid gerado pelo script 10 em out_dir e movê-lo para run_dir/000X.mid
    produced_midi: Optional[Path] = None
    if ok:
        # candidatos: .mid no out_dir criados/modificados após t0-3s (folga)
        candidates = []
        for p in Path(out_dir).glob("*.mid"):
            try:
                if p.stat().st_mtime >= (t0 - 3):
                    candidates.append(p)
            except Exception:
                pass
        if not candidates:
            # fallback: pega o mais novo de qualquer jeito
            candidates = sorted(Path(out_dir).glob("*.mid"), key=lambda x: x.stat().st_mtime, reverse=True)

        if candidates:
            produced_midi = max(candidates, key=lambda x: x.stat().st_mtime)

            # move/renomeia para o nome sequencial da run
            try:
                # se já existe, substitui
                if midi_path.exists():
                    midi_path.unlink()
                produced_midi.replace(midi_path)
            except Exception:
                # se não der para mover, tenta copiar
                try:
                    data = produced_midi.read_bytes()
                    midi_path.write_bytes(data)
                except Exception:
                    pass
        else:
            # nada encontrado no out_dir
            produced_midi = None



    # Render WAV (opcional)
    wav_bytes = None
    if ok and midi_path.exists() and soundfont and soundfont.exists():
        wav_bytes = _render_wav(midi_path, wav_path, soundfont)


    return {
        "ok": ok,
        "midi_path": midi_path if midi_path.exists() else None,
        "wav_path": wav_path if wav_path.exists() else None,
        "wav_bytes": wav_bytes,
        "stdout": stdout,
        "stderr": stderr,
        "cmd": cmd,
        "meta": {
            "prompt_text": prompt_text,
            "bpm": bpm,
            "time_signature": time_signature,
            # "duration_bars": duration_bars,
            "duration_seconds": duration_seconds,
            "temperature": temperature,
            "top_p": top_p,
            "allow_programs": allow_programs,
            "ban_drums": ban_drums,
            "force_program": force_program,
            "primer_bars": primer_bars,
            "primer_tokens": primer_tokens,
            "run_dir": str(run_dir),
            "seq_id": seq_id,
        }
    }
