# adapters/edit12_adapter.py
# Adapter para o script 12_edit_sections.py (edição de trecho)
# - Seleção temporal em segundos
# - Segue o mesmo formato/fluxo de continue_v11
# - WAV gerado com sufixo _edited.wav

from __future__ import annotations
import sys, subprocess, time
from pathlib import Path
from typing import Optional, Dict, Any
import re

from adapters.run_utils import next_seq_id, seq_paths
from adapters.gen10_adapter import _render_wav  # reaproveita renderer (pretty_midi/fluidsynth)

def edit_v12(
    *,
    # caminhos do projeto
    repo_root: Path,                 # raiz do repo (ex.: Path(__file__).resolve().parents[2])
    out_dir: Path,                   # base de saídas (ex.: repo_root / "data/infer_out")
    run_dir: Path,                   # pasta da run atual (onde ficam 000X.mid/.wav)
    script12: Optional[Path] = None, # caminho do src/12_edit_sections.py (resolve se None)

    # seleção temporal (obrigatória em segundos)
    start_seconds: float = 10,
    end_seconds: float = 15,

    # fonte
    source_midi: Path,               # MIDI a ser editado

    # contexto (opcional, em barras — repassado ao script 12)
    context_bars_before: int = 4,
    context_bars_after: int = 2,

    # parâmetros criativos essenciais (iguais aos de 10)
    bin_dir: Path,
    soft_ckpt: Path,
    ctrl_vocab: Path,
    prompt_text: str = "",
    tempo_bpm: int = 120,
    time_signature: str = "4/4",
    temperature: float = 1.2,
    top_p: float = 0.95,
    max_new_tokens: int = 5000,
    allow_programs: str = "",
    ban_drums: bool = False,

    # renderização (opcional)
    soundfont: Optional[Path] = None,

    # controle de nome (opcional)
    seq_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Executa o script 12 e retorna um dict no mesmo formato dos outros adapters.
    - Salva o MIDI final como run_dir/000X.mid
    - Renderiza o WAV como run_dir/000X_edited.wav
    """
    repo_root = Path(repo_root)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(run_dir); run_dir.mkdir(parents=True, exist_ok=True)

    script12 = script12 or (repo_root / "src" / "12_edit_sections.py")
    assert script12.exists(), f"script não encontrado: {script12}"
    assert source_midi.exists(), f"source_midi não encontrado: {source_midi}"
    assert bin_dir.exists(), f"bin_dir não encontrado: {bin_dir}"
    assert soft_ckpt.exists(), f"soft_ckpt não encontrado: {soft_ckpt}"
    assert ctrl_vocab.exists(), f"ctrl_vocab não encontrado: {ctrl_vocab}"

    # nomeia saídas da run
    seq_id = seq_id or next_seq_id(run_dir)
    midi_path, wav_path_base = seq_paths(run_dir, seq_id)  # run_dir/000X.mid, run_dir/000X.wav

    # >>> NOVO: criar tag do arquivo base
    base_tag = re.sub(r'[^A-Za-z0-9._-]+', '-', Path(source_midi).stem)

    # >>> NOVO: nomes finais "autoexplicativos"
    midi_named = midi_path.with_name(f"{seq_id}__base-{base_tag}_edited.mid")
    wav_named  = wav_path_base.with_name(f"{seq_id}__base-{base_tag}_edited.wav")

    # nome temporário que o script 12 vai gerar dentro de out_dir
    out_name = f"edited_{int(time.time())}_{seq_id}__base-{base_tag}.mid"

    # saneamento da seleção temporal
    s0 = max(0.0, float(start_seconds))
    e0 = max(s0 + 0.001, float(end_seconds))

    # monta comando conforme argparse do 12
    cmd = [
        sys.executable, str(script12),
        "--start_seconds", f"{s0:.6f}",
        "--end_seconds",   f"{e0:.6f}",
        "--source_midi",   str(source_midi),
        "--context_bars_before", str(int(context_bars_before)),
        "--context_bars_after",  str(int(context_bars_after)),

        "--out_dir",  str(out_dir),
        "--out_name", str(out_name),

        "--bin_dir",    str(bin_dir),
        "--soft_ckpt",  str(soft_ckpt),
        "--ctrl_vocab", str(ctrl_vocab),
        "--prompt_text", str(prompt_text or ""),
        "--tempo_bpm", str(int(tempo_bpm)),
        "--time_signature", str(time_signature),
        "--temperature", str(float(temperature)),
        "--top_p", str(float(top_p)),
        "--max_new_tokens", str(int(max_new_tokens)),
    ]
    if allow_programs:
        cmd += ["--allow_programs", allow_programs]
    if ban_drums:
        cmd += ["--ban_drums"]

    t0 = time.time()
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

    # localizar o .mid produzido no out_dir (preferir out_name)
    produced_midi: Optional[Path] = None
    if ok:
        # 1) caminho explícito que passamos ao script (--out_dir + --out_name)
        candidate = Path(out_dir) / out_name
        if candidate.exists():
            produced_midi = candidate
        else:
            # 2) fallback: tenta achar por padrão (caso o script ignore --out_name)
            #    exemplos de nomes: edited_<ts>_<seq_id>__base-<tag>.mid  OU  <seq_id>__base-<tag>.mid
            pat1 = f"*{seq_id}__base-{base_tag}.mid"
            found = sorted(Path(out_dir).glob(pat1))
            if found:
                produced_midi = found[-1]

    if ok and produced_midi and produced_midi.exists():
        try:
            if midi_named.exists():
                midi_named.unlink()
            produced_midi.replace(midi_named)
        except Exception:
            try:
                midi_named.write_bytes(produced_midi.read_bytes())
            except Exception as e:
                ok = False
                stderr += f"\n[edit12_adapter] Falha ao gravar MIDI final: {e}"
    else:
        if ok:
            # processo retornou 0 mas não achei o arquivo
            ok = False
            stderr += "\n[edit12_adapter] OK do processo, mas nenhum .mid foi encontrado em out_dir."

    # render como 000X_edited.wav
    wav_bytes = None
    if ok and midi_named.exists() and soundfont and Path(soundfont).exists():
        try:
            wav_bytes = _render_wav(midi_named, wav_named, soundfont)
        except Exception as e:
            stderr += f"\n[edit12_adapter] Render WAV falhou: {e}"

    return {
        "ok": ok,
        "midi_path": midi_named if ok and midi_named.exists() else None,
        "wav_path":  wav_named  if ok and wav_named.exists()  else None,
        "wav_bytes": wav_bytes,
        "stdout": stdout,
        "stderr": stderr,
        "cmd": cmd,
        "meta": {
            "seq_id": seq_id,
            "base_tag": base_tag,        # <<< útil para exibir/logar
            "start_seconds": s0,
            "end_seconds": e0,
            "source_midi": str(source_midi),
            "context_bars_before": context_bars_before,
            "context_bars_after": context_bars_after,
            "tempo_bpm": tempo_bpm,
            "time_signature": time_signature,
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
            "allow_programs": allow_programs,
            "ban_drums": ban_drums,
            "run_dir": str(run_dir),
        }
    }
