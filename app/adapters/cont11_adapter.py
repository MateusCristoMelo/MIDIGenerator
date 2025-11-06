# adapters/cont11_adapter.py
# Adapter para o script 11_infer_continue_softprompt.py (continuação)
# - Espelha a estrutura de gen10_adapter.py
# - Usa sys.executable, cwd=repo_root, timestamp, next_seq_id/seq_paths
# - Aceita duração em segundos (duration_seconds) como input
# - (Opcional) renderiza WAV via _render_wav importado do gen10_adapter
# - Exporta também o "trecho adicionado" (added_only)

from __future__ import annotations
import sys, subprocess, time
from pathlib import Path
from typing import Optional, Dict, Any
import re

import pretty_midi

from adapters.run_utils import next_seq_id, seq_paths
from adapters.gen10_adapter import _render_wav  # reaproveita o renderer

def _extract_added_only(final_mid: Path, base_mid: Path, out_added_mid: Path) -> bool:
    """
    Extrai do MIDI final somente o trecho posterior ao fim do MIDI base
    e salva em 'out_added_mid'. Retorna True se conseguiu.
    """
    try:
        pm_final = pretty_midi.PrettyMIDI(str(final_mid))
        pm_base  = pretty_midi.PrettyMIDI(str(base_mid))
        offset = pm_base.get_end_time()

        pm_added = pretty_midi.PrettyMIDI(resolution=pm_final.resolution)
        for inst in pm_final.instruments:
            new_inst = pretty_midi.Instrument(program=inst.program, is_drum=inst.is_drum, name=inst.name)
            # notas
            for n in inst.notes:
                if n.end <= offset:
                    continue
                ns = max(n.start, offset) - offset
                ne = n.end - offset
                if ne > ns + 1e-6:
                    new_inst.notes.append(pretty_midi.Note(n.velocity, n.pitch, ns, ne))
            # CC
            for c in inst.control_changes:
                if c.time >= offset:
                    new_inst.control_changes.append(pretty_midi.ControlChange(c.number, c.value, c.time - offset))
            # PB
            for b in inst.pitch_bends:
                if b.time >= offset:
                    new_inst.pitch_bends.append(pretty_midi.PitchBend(b.pitch, b.time - offset))

            if new_inst.notes or new_inst.control_changes or new_inst.pitch_bends:
                pm_added.instruments.append(new_inst)

        out_added_mid.parent.mkdir(parents=True, exist_ok=True)
        pm_added.write(str(out_added_mid))
        return True
    except Exception as e:
        print(f"[cont11_adapter] Falha ao extrair trecho adicionado: {e}")
        return False


def continue_v11(
    *,
    # caminhos do projeto
    repo_root: Path,             # raiz do projeto (ex.: Path(__file__).resolve().parents[2])
    bin_dir: Path,               # ex.: repo_root / "data/binpack"
    soft_ckpt: Path,             # ex.: repo_root / "runs/softprompt_midicaps/softprompt_ep02.pt"
    ctrl_vocab: Path,            # ex.: repo_root / "data/ctrl_vocab.json"
    out_dir: Path,               # ex.: repo_root / "data/infer_out"
    run_dir: Path,               # diretório da run atual (onde ficará 000X.mid/.wav)
    script11: Optional[Path] = None,  # caminho do src/11_infer_continue_softprompt.py (resolvido se None)

    # insumo base
    source_midi: Path,           # MIDI a ser continuado

    # parâmetros musicais
    tempo_bpm: int = 120,
    time_signature: str = "4/4",
    duration_seconds: float = 30.0,   # duração a gerar (segundos) — exigido pelo script 11
    tail_bars: int = 8,               # barras finais do source usadas como primer (obrigatório no 11)

    # sampling
    temperature: float = 1.2,
    top_p: float = 0.95,
    top_k: int = 0,
    max_new_tokens: int = 3000,

    # restrições
    allow_programs: str = "",   # "40,41,42,..." (GM)
    ban_drums: bool = False,
    rest_penalty: float = 0.0,
    timeshift_penalty: float = 0.0,

    # prompt textual (opcional)
    prompt_text: str = "",

    # renderização (opcional)
    soundfont: Optional[Path] = None,

    # controle de nome
    seq_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Executa o script 11 e retorna caminhos/bytes gerados (no padrão do gen10_adapter).

    Retorno:
      {
        "ok": bool,
        "midi_path": Path|None,          # 000X.mid (final)
        "wav_path": Path|None,           # 000X.wav
        "wav_bytes": bytes|None,
        "added_mid_path": Path|None,     # 000X_added.mid
        "added_wav_path": Path|None,     # 000X_added.wav
        "stdout": str,
        "stderr": str,
        "cmd": list[str],
        "meta": {...}
      }
    """
    # Resolve caminho do script 11
    script11 = script11 or (repo_root / "src" / "11_infer_continue_softprompt.py")
    assert script11.exists(), f"script não encontrado: {script11}"
    assert bin_dir.exists(), f"bin_dir não encontrado: {bin_dir}"
    assert soft_ckpt.exists(), f"soft_ckpt não encontrado: {soft_ckpt}"
    assert ctrl_vocab.exists(), f"ctrl_vocab não encontrado: {ctrl_vocab}"
    assert source_midi.exists(), f"source_midi não encontrado: {source_midi}"

    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    # ID/paths da run
    seq_id = seq_id or next_seq_id(run_dir)
    midi_path, wav_path = seq_paths(run_dir, seq_id)

    # 1) crie uma tag segura a partir do arquivo base
    base_tag = re.sub(r'[^A-Za-z0-9._-]+', '-', Path(source_midi).stem)  # ex.: "0005"

    # 2) redefina os nomes finais com a tag do base
    midi_named  = midi_path.with_name(f"{seq_id}__base-{base_tag}.mid")
    wav_named   = wav_path.with_name(f"{seq_id}__base-{base_tag}.wav")
    added_mid   = midi_path.with_name(f"{seq_id}__base-{base_tag}_added.mid")
    added_wav   = wav_path.with_name(f"{seq_id}__base-{base_tag}_added.wav")

    # added_mid = midi_path.with_name(f"{midi_path.stem}_added.mid")
    # added_wav = wav_path.with_name(f"{wav_path.stem}_added.wav")

    # Nome que o script 11 vai gerar dentro de out_dir (depois movemos)
    out_name = f"cont_{int(time.time())}_{seq_id}.mid"

    t0 = time.time()

    # Monta comando (alinhado ao argparse do 11)
    cmd = [
        sys.executable, str(script11),
        "--bin_dir", str(bin_dir),
        "--soft_ckpt", str(soft_ckpt),
        "--ctrl_vocab", str(ctrl_vocab),
        "--out_dir", str(out_dir),
        "--out_name", out_name,

        "--prompt_text", str(prompt_text or ""),
        "--tempo_bpm", str(int(tempo_bpm)),
        "--time_signature", str(time_signature),
        "--duration_seconds", f"{float(duration_seconds):.6f}",

        "--temperature", str(float(temperature)),
        "--top_p", str(float(top_p)),
        "--top_k", str(int(top_k)),
        "--max_new_tokens", str(int(max_new_tokens)),

        "--source_midi", str(source_midi),
        "--tail_bars", str(int(tail_bars)),
    ]

    if allow_programs:
        cmd += ["--allow_programs", allow_programs]
    if ban_drums:
        cmd += ["--ban_drums"]
    if rest_penalty:
        cmd += ["--rest_penalty", str(float(rest_penalty))]
    if timeshift_penalty:
        cmd += ["--timeshift_penalty", str(float(timeshift_penalty))]

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

    produced_midi: Optional[Path] = None
    if ok:
        # Tenta pegar exatamente o out_name:
        candidate = out_dir / out_name
        if candidate.exists():
            produced_midi = candidate
        else:
            # fallback: .mid alterados recentemente
            fresh = []
            for p in Path(out_dir).glob("*.mid"):
                try:
                    if p.stat().st_mtime >= (t0 - 3):
                        fresh.append(p)
                except Exception:
                    pass
            if fresh:
                produced_midi = max(fresh, key=lambda x: x.stat().st_mtime)

    # mover/renomear o .mid gerado pelo script 11
    if ok and produced_midi:
        try:
            if midi_named.exists():
                midi_named.unlink()
            produced_midi.replace(midi_named)
        except Exception:
            try:
                data = produced_midi.read_bytes()
                midi_named.write_bytes(data)
            except Exception:
                ok = False
                stderr += f"\n[cont11_adapter] Falhou mover/copiar MIDI para {midi_named}"


    # Render WAV (opcional)
    wav_bytes = None
    if ok and midi_named.exists() and soundfont and Path(soundfont).exists():
        wav_bytes = _render_wav(midi_named, wav_named, soundfont)

    # gerar "added only"
    added_mid_path = None
    added_wav_path = None
    if ok and midi_named.exists():
        if _extract_added_only(midi_named, source_midi, added_mid):
            added_mid_path = added_mid
            if soundfont and Path(soundfont).exists():
                try:
                    _ = _render_wav(added_mid, added_wav, soundfont)
                    if added_wav.exists():
                        added_wav_path = added_wav
                except Exception as e:
                    print(f"[cont11_adapter] Falha ao renderizar added WAV: {e}")


    # Gera "somente trecho adicionado" + render
    added_mid_path = None
    added_wav_path = None
    if ok and midi_path.exists():
        if _extract_added_only(midi_path, source_midi, added_mid):
            added_mid_path = added_mid
            if soundfont and Path(soundfont).exists():
                try:
                    _ = _render_wav(added_mid, added_wav, soundfont)
                    if added_wav.exists():
                        added_wav_path = added_wav
                except Exception as e:
                    print(f"[cont11_adapter] Falha ao renderizar added WAV: {e}")

    return {
        "ok": ok,
        "midi_path": midi_named if ok and midi_named.exists() else None,
        "wav_path":  wav_named  if ok and wav_named.exists()  else None,
        "wav_bytes": wav_bytes,
        "added_mid_path": added_mid_path if added_mid_path and added_mid_path.exists() else None,
        "added_wav_path": added_wav_path if added_wav_path and added_wav_path.exists() else None,
        "stdout": stdout,
        "stderr": stderr,
        "cmd": cmd,
        "meta": {
            "source_midi": str(source_midi),
            "bpm": tempo_bpm,
            "time_signature": time_signature,
            "duration_seconds": duration_seconds,
            "tail_bars": tail_bars,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_new_tokens": max_new_tokens,
            "allow_programs": allow_programs,
            "ban_drums": ban_drums,
            "rest_penalty": rest_penalty,
            "timeshift_penalty": timeshift_penalty,
            "run_dir": str(run_dir),
            "seq_id": seq_id,
            "base_tag": base_tag,
        }
    }
