# adapters/midi_analyzer.py
from __future__ import annotations
import io, csv, json, math
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pretty_midi as pm

class MidiAnalyzer:
    """Adapter de análise de MIDI para o TG.
    Sem dependência do Streamlit (para poder testar em CLI)."""

    # --------- API principal ---------
    def analyze(self, midi_path: Path) -> Dict[str, Any]:
        pm_obj = self._load_midi(midi_path)
        total_sec = float(pm_obj.get_end_time())
        tempo_bpm, (ts_num, ts_den), tempo_profile = self._tempo_ts(pm_obj)
        bars = self._bars_estimate(total_sec, tempo_bpm or 120.0, ts_num, ts_den)
        poly = self._polyphony_profile(pm_obj)
        vel  = self._note_velocity_stats(pm_obj)
        prng = self._pitch_range(pm_obj)
        pch  = self._pitch_class_histogram(pm_obj)
        prog = self._program_palette(pm_obj)
        note_count = int(sum(len(i.notes) for i in pm_obj.instruments))

        return {
            "file": str(midi_path),
            "duration_sec": total_sec,
            "duration_hhmmss": self._sec_to_mmss(total_sec),
            "tempo_bpm_median": tempo_bpm,
            "tempo_profile": tempo_profile,       # [{t,bpm}]
            "time_signature": {"numerator": ts_num, "denominator": ts_den},
            "bars_estimated": bars,
            "notes_total": note_count,
            "polyphony": poly,                    # mean/p95/max
            "velocity_stats": vel,                # mean/std/min/max
            "pitch_range": prng,                  # min/max/span
            "pitch_class_histogram": pch,         # 12 bins (C..B)
            "program_palette": prog,              # GM + drums
        }

    # --------- Utilidades públicas ---------
    def sibling_wav_for(self, midi_path: Path) -> Optional[Path]:
        candidates = list(Path(midi_path).parent.glob(Path(midi_path).stem + "*.wav"))
        return candidates[0] if candidates else None

    def metrics_to_csv_bytes(self, metrics: Dict[str, Any]) -> bytes:
        flat = {}
        for k, v in metrics.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    flat[f"{k}.{kk}"] = vv
            else:
                flat[k] = v if not isinstance(v, (list, tuple)) else json.dumps(v)
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=list(flat.keys()))
        writer.writeheader()
        writer.writerow(flat)
        return buf.getvalue().encode("utf-8")

    # --------- Privados (helpers) ---------
    def _load_midi(self, midi_path: Path) -> pm.PrettyMIDI:
        with open(midi_path, "rb") as f:
            return pm.PrettyMIDI(io.BytesIO(f.read()))

    def _sec_to_mmss(self, t: float) -> str:
        t = max(0.0, float(t))
        m = int(t // 60)
        s = int(round(t - 60*m))
        return f"{m}:{s:02d}"

    def _bars_estimate(self, total_seconds: float, tempo_bpm: float, ts_num: int, ts_den: int) -> Optional[int]:
        if tempo_bpm <= 0 or ts_num <= 0:
            return None
        beats = (tempo_bpm / 60.0) * total_seconds
        beats_per_bar = ts_num * (4.0/ts_den)
        if beats_per_bar <= 0:
            return None
        return max(1, int(round(beats / beats_per_bar)))

    def _pitch_class_histogram(self, pm_obj: pm.PrettyMIDI) -> List[float]:
        hist = np.zeros(12, dtype=float)
        total = 0.0
        for inst in pm_obj.instruments:
            for n in inst.notes:
                dur = max(1e-3, n.end - n.start)
                hist[n.pitch % 12] += dur
                total += dur
        if total > 0:
            hist /= total
        return hist.tolist()

    def _polyphony_profile(self, pm_obj: pm.PrettyMIDI, fps: int = 100) -> Dict[str, float]:
        if pm_obj.get_end_time() <= 0:
            return {"mean":0.0, "p95":0.0, "max":0}
        T = pm_obj.get_end_time()
        ts = np.linspace(0, T, max(2, int(T*fps)))
        poly = np.zeros_like(ts, dtype=int)
        for inst in pm_obj.instruments:
            for n in inst.notes:
                i0 = int((n.start / T) * (len(ts)-1)) if T > 0 else 0
                i1 = int((n.end   / T) * (len(ts)-1)) if T > 0 else 0
                if i1 <= i0: i1 = min(len(ts)-1, i0+1)
                poly[i0:i1] += 1
        return {
            "mean": float(np.mean(poly)),
            "p95":  float(np.percentile(poly, 95)),
            "max":  int(np.max(poly)),
        }

    def _note_velocity_stats(self, pm_obj: pm.PrettyMIDI) -> Dict[str, Optional[float]]:
        vels = []
        for inst in pm_obj.instruments:
            for n in inst.notes:
                vels.append(n.velocity)
        if not vels:
            return {"mean": None, "std": None, "min": None, "max": None}
        v = np.array(vels, dtype=float)
        return {"mean": float(v.mean()), "std": float(v.std()), "min": int(v.min()), "max": int(v.max())}

    def _pitch_range(self, pm_obj: pm.PrettyMIDI) -> Dict[str, Optional[int]]:
        pitches = []
        for inst in pm_obj.instruments:
            for n in inst.notes:
                pitches.append(n.pitch)
        if not pitches:
            return {"min": None, "max": None, "span": None}
        return {"min": int(min(pitches)), "max": int(max(pitches)), "span": int(max(pitches)-min(pitches))}

    def _program_palette(self, pm_obj: pm.PrettyMIDI) -> Dict[str, Any]:
        progs = []
        drums = False
        for inst in pm_obj.instruments:
            if inst.is_drum:
                drums = True
            else:
                progs.append(inst.program)
        return {"programs": sorted(list(set(progs))), "has_drums": bool(drums)}

    def _tempo_ts(self, pm_obj: pm.PrettyMIDI) -> Tuple[Optional[float], Tuple[int,int], list]:
        tempo_times, tempi = pm_obj.get_tempo_changes()
        if len(tempi) == 0:
            tempo_bpm = None
            tempo_profile = []
        else:
            tempo_bpm = float(np.median(tempi))
            tempo_profile = [{"t": float(t), "bpm": float(b)} for t, b in zip(tempo_times, tempi)]
        tsc = pm_obj.time_signature_changes
        if len(tsc) == 0:
            ts_num, ts_den = 4, 4
        else:
            ts_num, ts_den = tsc[-1].numerator, tsc[-1].denominator
        return tempo_bpm, (ts_num, ts_den), tempo_profile
