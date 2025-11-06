# src/adapters/anal13_adapter.py
# Adapter da etapa 13 – análise de métricas (v13)

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import pretty_midi as pm
import numpy as np

def melodic_variety(pmid: pm.PrettyMIDI) -> float:
    hist = np.zeros(128)
    for inst in pmid.instruments:
        for n in inst.notes:
            hist[n.pitch] += 1
    used = np.count_nonzero(hist)
    return used / 128.0  # fração de notas possíveis usadas

def temporal_coherence(pmid: pm.PrettyMIDI) -> float:
    starts = sorted([n.start for i in pmid.instruments for n in i.notes])
    if len(starts) < 3: return 1.0
    intervals = np.diff(starts)
    return 1.0 / (np.std(intervals) + 1e-6)  # quanto menor a variação, maior a coerência

def loopability_score(pmid: pm.PrettyMIDI, window_s: float = 4.0) -> float:
    """
    Mede a similaridade entre o início e o final da música, como proxy de loopabilidade.
    Retorna valor ∈ [0, 1], onde 1 = encaixe perfeito.
    """
    dur = pmid.get_end_time()
    if dur <= 2 * window_s:
        return 0.0  # muito curta p/ análise

    # função auxiliar: pitch-class histograma ponderado por duração das notas em um intervalo
    def pc_hist_in_range(start_t: float, end_t: float) -> np.ndarray:
        pc = np.zeros(12, dtype=float)
        for inst in pmid.instruments:
            for n in inst.notes:
                if n.end <= start_t or n.start >= end_t:
                    continue
                overlap = max(0.0, min(n.end, end_t) - max(n.start, start_t))
                pc[n.pitch % 12] += overlap
        if pc.sum() > 0:
            pc /= pc.sum()
        return pc

    start_hist = pc_hist_in_range(0.0, window_s)
    end_hist   = pc_hist_in_range(dur - window_s, dur)

    # similaridade por cosseno (entre 0 e 1)
    num = np.dot(start_hist, end_hist)
    den = np.linalg.norm(start_hist) * np.linalg.norm(end_hist)
    sim = num / (den + 1e-9)
    return float(np.clip(sim, 0.0, 1.0))

def first_time_signature(pmid: pm.PrettyMIDI) -> Tuple[int, int]:
    if pmid.time_signature_changes:
        ts = pmid.time_signature_changes[0]
        return ts.numerator, ts.denominator
    return 4, 4

def bar_boundaries(pmid: pm.PrettyMIDI) -> List[float]:
    """Aproxima início de cada compasso a partir de beats e TS inicial."""
    num, den = first_time_signature(pmid)
    bpb = max(1, int(round(num * (4.0 / den))))
    beats = pmid.get_beats()
    if len(beats) == 0:
        dur = pmid.get_end_time()
        if dur <= 0:
            return [0.0]
        # fallback: 2 s por compasso
        grid = list(np.arange(0.0, dur + 1e-9, 2.0))
        if grid[-1] < dur:
            grid.append(dur)
        return grid
    bars = [beats[i] for i in range(0, len(beats), bpb)]
    if not bars or bars[0] > 1e-6:
        bars = [0.0] + bars
    end_t = pmid.get_end_time()
    if bars[-1] < end_t:
        bars.append(end_t)
    return bars

def tempo_changes_to_bpm(pmid: pm.PrettyMIDI) -> np.ndarray:
    """Extrai vetor de BPM a partir de mudanças de tempo. Se não houver, assume 120."""
    tempos, times = pmid.get_tempo_changes()
    if tempos is None or len(tempos) == 0:
        return np.array([120.0])
    # tempos já vêm em BPM (pretty_midi retorna tempos em BPM)
    return tempos


# ---------------------------
# Polifonia e densidade
# ---------------------------

def polyphony_stats(notes: List[pm.Note]) -> Tuple[float, int]:
    """
    Calcula polifonia média e máxima varrendo onsets/offsets.
    Complexidade O(N log N). Retorna (poly_avg, poly_max).
    """
    if not notes:
        return 0.0, 0
    events = []
    for n in notes:
        events.append((n.start, +1))
        events.append((n.end,   -1))
    events.sort(key=lambda x: (x[0], -x[1]))  # nota que termina não conta como sobreposição com a que começa no mesmo t

    active = 0
    last_t = events[0][0]
    area = 0.0
    poly_max = 0
    end_time = max((n.end for n in notes), default=last_t)

    for t, delta in events:
        if t > last_t:
            area += active * (t - last_t)
            last_t = t
        active += delta
        if active > poly_max:
            poly_max = active

    duration = max(1e-9, end_time - events[0][0])
    poly_avg = area / duration
    return float(poly_avg), int(poly_max)

import mido

def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cw = np.cumsum(w / (w.sum() + 1e-12))
    return float(v[np.searchsorted(cw, 0.5)])

def bpm_stats_meta(src_path: Path) -> tuple[float, float, float] | None:
    """(min, mediana ponderada por duração, max) a partir de set_tempo via mido."""
    try:
        mid = mido.MidiFile(src_path)
        tpb = mid.ticks_per_beat
        tempo_uspb = 500_000  # 120 bpm padrão
        cur_bpm = 60_000_000 / tempo_uspb

        merged = mido.merge_tracks(mid.tracks)
        seg_bpms: list[float] = []
        seg_durs: list[float] = []
        acc_sec = 0.0

        for msg in merged:
            if msg.time:
                acc_sec += mido.tick2second(msg.time, tpb, tempo_uspb)
            if msg.is_meta and msg.type == "set_tempo":
                if acc_sec > 1e-9:
                    seg_bpms.append(cur_bpm)
                    seg_durs.append(acc_sec)
                    acc_sec = 0.0
                tempo_uspb = max(1, int(msg.tempo))
                cur_bpm = 60_000_000 / tempo_uspb

        if acc_sec > 1e-9:
            seg_bpms.append(cur_bpm)
            seg_durs.append(acc_sec)

        if not seg_bpms:
            return None

        bpms = np.asarray(seg_bpms, dtype=float)
        durs = np.asarray(seg_durs, dtype=float)
        # saneamento leve (descarta valores absurdos)
        keep = (bpms > 5.0) & np.isfinite(bpms)
        if not np.any(keep):
            return None
        bpms = bpms[keep]; durs = durs[keep]

        return float(bpms.min()), _weighted_median(bpms, durs), float(bpms.max())
    except Exception:
        return None
    
def bpm_choose(src_path: Path, pmid: pm.PrettyMIDI) -> tuple[float, float, float, str]:
    """
    Escolhe entre meta (mido) e estimate (pretty_midi) com regras simples.
    Retorna (bpm_min, bpm_med, bpm_max, source_tag).
    """
    meta = bpm_stats_meta(src_path)
    # pretty_midi.estimate_tempo -> único valor
    try:
        est = float(pmid.estimate_tempo())
        est_ok = np.isfinite(est) and (5.0 < est < 400.0)
    except Exception:
        est, est_ok = 120.0, False

    # 1) se meta existir e estiver em faixa razoável e consistente, use meta
    if meta is not None:
        mn, md, mx = meta
        meta_in_range = (30.0 <= md <= 240.0)
        meta_consistent = (mx / max(1e-6, mn)) <= 1.4  # variação ≤ 40%
        if meta_in_range and meta_consistent:
            return mn, md, mx, "meta"

    # 2) senão, se estimate estiver bom, use estimate
    if est_ok and (30.0 <= est <= 240.0):
        return est, est, est, "estimate"

    # 3) se meta existe, use meta mesmo (sem clamp agressivo)
    if meta is not None:
        mn, md, mx = meta
        return mn, md, mx, "meta_unsafe"

    # 4) fallback
    return 120.0, 120.0, 120.0, "fallback"
    
def note_density(notes: List[pm.Note], duration_seconds: float) -> float:
    if duration_seconds <= 0.0:
        return 0.0
    return float(len(notes)) / float(duration_seconds)

# ---------------------------
# Tom/Modo (estimativa simples - Krumhansl)
# ---------------------------

KH_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                     2.52, 5.19, 2.39, 3.66, 2.29, 2.88], dtype=float)
KH_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                     2.54, 4.75, 3.98, 2.69, 3.34, 3.17], dtype=float)
PITCH_CLASS_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

def estimate_key(pmid: pm.PrettyMIDI) -> Tuple[str, float]:
    """
    Estima tom e modo por histograma de classes de altura ponderado por duração das notas.
    Retorna (label, confidence) onde label = 'C major' / 'A minor' etc, e confidence ∈ [0,1].
    """
    pc = np.zeros(12, dtype=float)
    for inst in pmid.instruments:
        for n in inst.notes:
            d = max(1e-6, n.end - n.start)
            pc[n.pitch % 12] += d
    if pc.sum() <= 0:
        return "Unknown", 0.0

    pc = pc / (pc.sum() + 1e-9)
    # correlaciona com 12 transposições de major/minor
    best_label, best_score = "Unknown", -1e9
    best_norm = 1.0
    for rot in range(12):
        maj = np.roll(KH_MAJOR, rot)
        mino = np.roll(KH_MINOR, rot)
        sc_maj = np.dot(pc, maj)
        sc_min = np.dot(pc, mino)
        if sc_maj >= sc_min:
            label = f"{PITCH_CLASS_NAMES[rot]} major"
            score = sc_maj
            norm = np.linalg.norm(maj)
        else:
            label = f"{PITCH_CLASS_NAMES[rot]} minor"
            score = sc_min
            norm = np.linalg.norm(mino)
        if score > best_score:
            best_label, best_score, best_norm = label, score, norm

    # normaliza a "confiança" para ~[0..1]
    conf = float(best_score / (best_norm * np.linalg.norm(pc) + 1e-9))
    conf = float(np.clip(conf, 0.0, 1.0))
    return best_label, conf

# Importa funções do 13_analyze_metrics (estão no mesmo repo)
# Obs.: manter import "local" para evitar custo quando o adapter é importado sem uso.
def instrument_metrics(inst: pm.Instrument) -> Dict[str, Any]:
    notes = inst.notes
    is_drum = bool(inst.is_drum)
    program = int(inst.program)

    if notes:
        starts = np.array([n.start for n in notes], dtype=float)
        ends   = np.array([n.end   for n in notes], dtype=float)
        vels   = np.array([n.velocity for n in notes], dtype=float)
        pitches= np.array([n.pitch for n in notes], dtype=float)
        durs   = ends - starts

        dur_span = float(max(ends) - min(starts))
        dens = note_density(notes, max(1e-9, dur_span))
        poly_avg, poly_max = polyphony_stats(notes)

        return {
            "notes": int(len(notes)),
            "density_notes_per_s": float(dens),
            "polyphony_avg": float(poly_avg),
            "polyphony_max": int(poly_max),
            "pitch_min": int(pitches.min()),
            "pitch_max": int(pitches.max()),
            "pitch_mean": float(pitches.mean()),
            "velocity_mean": float(vels.mean()),
            "velocity_std": float(vels.std(ddof=0)),
            "note_duration_median_s": float(np.median(durs)),
            "span_duration_s": float(dur_span)
        }
    else:
        return {
            "notes": 0,
            "density_notes_per_s": 0.0,
            "polyphony_avg": 0.0,
            "polyphony_max": 0,
            "pitch_min": np.nan,
            "pitch_max": np.nan,
            "pitch_mean": np.nan,
            "velocity_mean": np.nan,
            "velocity_std": np.nan,
            "note_duration_median_s": np.nan,
            "span_duration_s": 0.0
        }

def global_metrics(pmid: pm.PrettyMIDI, src_path_global: Path) -> Dict[str, Any]:
    dur = float(pmid.get_end_time())
    tsn, tsd = first_time_signature(pmid)
    bars = bar_boundaries(pmid)
    bars_count = max(0, len(bars) - 1)

    # --- BPM robusto ---
    bpm_vals = None
    try:
        tempi, _ = pmid.get_tempo_changes()          # tempi em BPM (float)
        if tempi is not None and len(tempi) > 0 and np.all(np.isfinite(tempi)):
            bpm_vals = np.asarray(tempi, dtype=float)
    except Exception:
        bpm_vals = None

    if bpm_vals is None or bpm_vals.size == 0:
        # fallback: estima por espaçamento entre beats
        beats = pmid.get_beats()
        if beats is not None and len(beats) >= 2:
            diffs = np.diff(np.asarray(beats, dtype=float))
            diffs = diffs[diffs > 1e-6]
            if diffs.size > 0:
                bpm_vals = 60.0 / diffs

    if bpm_vals is None or bpm_vals.size == 0 or not np.all(np.isfinite(bpm_vals)):
        bpm_vals = np.array([120.0], dtype=float)    # último fallback

    # clamp mínimo para evitar zeros por arredondamento/ruído
    bpm_vals = np.clip(bpm_vals, 1.0, None)
    # t_min = float(np.min(bpm_vals))
    # t_med = float(np.median(bpm_vals))
    # t_max = float(np.max(bpm_vals))
    # --- BPM robusto (via mido + fallbacks) ---
    bpm_min, bpm_med, bpm_max, bpm_src = bpm_choose(src_path_global, pmid)

    # --- fim BPM robusto ---

    # flat list de notas
    all_notes, drum_notes, programs = [], 0, set()
    for inst in pmid.instruments:
        programs.add(int(inst.program))
        all_notes.extend(inst.notes)
        if inst.is_drum:
            drum_notes += len(inst.notes)

    poly_avg, poly_max = polyphony_stats(all_notes)
    dens = note_density(all_notes, max(1e-9, dur))
    key_label, key_conf = estimate_key(pmid)

    return {
        "duration_seconds": dur,
        "bars_count": int(bars_count),
        "time_signature": f"{tsn}/{tsd}",
        "bpm_min": bpm_min,
        "bpm_median": bpm_med,
        "bpm_max": bpm_max,
        "bpm_source": bpm_src,
        "notes_total": int(len(all_notes)),
        "density_notes_per_s": float(dens),
        "polyphony_avg": float(poly_avg),
        "polyphony_max": int(poly_max),
        "drum_notes_total": int(drum_notes),
        "instruments_count": int(len(pmid.instruments)),
        "programs_used": ",".join(str(p) for p in sorted(programs)),
        "estimated_key": key_label,
        "estimated_key_confidence": key_conf,
    }

def _analyze_single_midi(src_path: Path) -> pd.DataFrame:
    """Calcula as métricas (GLOBAL + por trilha) para um único .mid e devolve um DataFrame."""
    if not src_path.exists():
        raise FileNotFoundError(f"source_midi não encontrado: {src_path}")

    pmid = pm.PrettyMIDI(str(src_path))
    rows: List[Dict[str, Any]] = []

    # --- métricas globais principais ---
    g = global_metrics(pmid, src_path)

    # --- MÉTRICAS DO TG (novas) ---
    try:
        g["melodic_variety"] = float(melodic_variety(pmid))
        g["temporal_coherence"] = float(temporal_coherence(pmid))
        g["loopability"] = float(loopability_score(pmid))
    except Exception as e:
        print(f"[anal13_adapter] Falha ao calcular métricas TG: {e}")
        g["melodic_variety"] = None
        g["temporal_coherence"] = None
        g["loopability"] = None

    # --- bloco GLOBAL ---
    rows.append({
        "scope": "GLOBAL",
        "track_name": "",
        "program": 0,        # <- evitar tipo misto
        "is_drum": False,
        **g
    })

    # --- blocos por instrumento ---
    for i, inst in enumerate(pmid.instruments):
        m = instrument_metrics(inst)
        rows.append({
            "scope": "TRACK",
            "track_name": inst.name if inst.name else f"Track_{i:02d}",
            "program": int(inst.program),
            "is_drum": bool(inst.is_drum),
            **m
        })

    # --- DataFrame final (sanitizado) ---
    df = pd.DataFrame(rows)

    # ⚙️ Corrige tipos mistos (PyArrow exige tipo único)
    for col in ["program", "is_drum", "track_name", "scope"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # substitui NaN por None para evitar crash
    df = df.replace({np.nan: None})

    return df



def analyze_v13(
    *,
    source_midi: str | Path,
    out_dir: str | Path,
    out_name: str = "metrics.csv",
) -> Dict[str, Any]:
    """
    Roda a análise (v13) sobre um único arquivo MIDI.

    Parâmetros
    ----------
    source_midi : str | Path
        Caminho do .mid a analisar.
    out_dir : str | Path
        Pasta onde o CSV será salvo (criada se necessário).
    out_name : str
        Nome do arquivo CSV (padrão: 'metrics.csv').

    Retorno
    -------
    dict
        {
          "ok": bool,
          "csv_path": Path,
          "rows": int,
          "global": Dict[str, Any],       # linha GLOBAL (já como dict)
          "tracks": List[Dict[str, Any]], # linhas TRACK
          "dataframe": pd.DataFrame       # DF completo (útil para UI/tests)
        }
    """
    src = Path(source_midi)
    out_base = Path(out_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    out_csv = out_base / out_name

    df = _analyze_single_midi(src)

    # garante tipos estáveis no CSV
    df.to_csv(out_csv, index=False, float_format="%.6f")

    # quebra em global + tracks para retorno “amigável”
    global_row = df[df["scope"] == "GLOBAL"].iloc[0].to_dict() if (df["scope"] == "GLOBAL").any() else {}
    track_rows = df[df["scope"] == "TRACK"].to_dict(orient="records")

    return {
        "ok": True,
        "csv_path": out_csv,
        "rows": int(len(df)),
        "global": global_row,
        "tracks": track_rows,
        "dataframe": df,
    }
