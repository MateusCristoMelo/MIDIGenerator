# app.py
# Streamlit UI – Geração (10), Continuação (11), Edição (12)
# - Runs em /data/infer_out/run_YYYYMMDD_HHMMSS
# - Arquivos sequenciais 0001.mid/.wav, 0002.mid/.wav, ...
# - Aba Geração: lista todas as faixas com waveform + player (Wavesurfer.js)
# - Sem botão Redo na Geração

from __future__ import annotations
from pathlib import Path
import uuid, base64, sys, tempfile
import streamlit as st
import streamlit.components.v1 as components
# from streamlit_wavesurfer import wavesurfer  # Wavesurfer.js player + waveform
# from streamlit_wavesurfer.utils import WaveSurferOptions  # Wavesurfer.js player + waveform
from adapters.midi_analyzer import MidiAnalyzer
from typing import Dict, Any, List, Tuple
import pretty_midi as pm
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent / "template-reactless"))

from my_component import wavesurfer_region_selector

analyzer = MidiAnalyzer()

@st.cache_data(show_spinner=False)
def analyze_cached(midi_path_str: str):
    return analyzer.analyze(Path(midi_path_str))

def wavesurfer_player(wav_path: Path, title="Track"):
    uid = f"ws-{uuid.uuid4().hex}"
    b64 = base64.b64encode(Path(wav_path).read_bytes()).decode()
    data_url = f"data:audio/wav;base64,{b64}"
    html = f"""
    <div id="{uid}" style="border:1px solid #e5e7eb;padding:12px;border-radius:12px;">
      <div style="font:600 14px system-ui;margin-bottom:6px">{title}</div>
      <div id="{uid}-wave"></div>
      <div style="display:flex;align-items:center;gap:12px;margin-top:6px">
        <button id="{uid}-play" style="padding:6px 10px;border-radius:8px;border:1px solid #d1d5db;background:#fff;cursor:pointer">▶️ Play</button>
        <span id="{uid}-time" style="font:500 12px system-ui;color:#374151">0:00 / 0:00</span>
      </div>
    </div>
    <script src="https://unpkg.com/wavesurfer.js@7"></script>
    <script>
      (function() {{
        const fmt = s => {{
          s = Math.max(0, s|0);
          const m = (s/60)|0, r = (s%60)|0;
          return m + ":" + String(r).padStart(2, '0');
        }};
        const wave = WaveSurfer.create({{
          container: '#{uid}-wave',
          url: '{data_url}',
          height: 72,
          waveColor: '#8aa',
          progressColor: '#4c8',
          cursorColor: '#111',
          barWidth: 2,
          barGap: 1,
          normalize: true,
          backend: 'MediaElement',   // garante timer nativo
        }});
        const playBtn = document.getElementById('{uid}-play');
        const timeLbl = document.getElementById('{uid}-time');
        let duration = 0;

        const updateTime = () => {{
          const cur = wave.getCurrentTime();
          timeLbl.textContent = fmt(cur) + ' / ' + fmt(duration);
        }};

        wave.on('ready', () => {{
          duration = wave.getDuration();
          updateTime();
        }});
        wave.on('audioprocess', updateTime);
        wave.on('seek', updateTime);
        wave.on('pause', updateTime);
        wave.on('finish', updateTime);

        playBtn.addEventListener('click', () => {{
          wave.playPause();
          playBtn.textContent = wave.isPlaying() ? '⏸️ Pause' : '▶️ Play';
        }});
      }})();
    </script>
    """
    components.html(html, height=170, scrolling=False)


# ---------- helpers ----------
def _human_hhmmss(seconds: float) -> str:
    s = int(round(max(0.0, seconds)))
    h = s // 3600
    m = (s % 3600) // 60
    s = s % 60
    return f"{h:d}:{m:02d}:{s:02d}"

def _pitch_class_histogram(pmid: pm.PrettyMIDI) -> List[float]:
    pc = np.zeros(12, dtype=float)
    for inst in pmid.instruments:
        for n in inst.notes:
            d = max(1e-6, n.end - n.start)
            pc[n.pitch % 12] += d
    if pc.sum() <= 0: 
        return [0.0]*12
    pc = (pc / pc.sum()).tolist()
    return pc

def _bar_boundaries(pmid: pm.PrettyMIDI) -> List[float]:
    # mesma heurística do 13 (beats + TS inicial)
    if pmid.time_signature_changes:
        ts = pmid.time_signature_changes[0]
        num, den = ts.numerator, ts.denominator
    else:
        num, den = 4, 4
    bpb = max(1, int(round(num * (4.0 / den))))
    beats = pmid.get_beats()
    if len(beats) == 0:
        dur = pmid.get_end_time()
        if dur <= 0: 
            return [0.0]
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

def _polyphony_profile(notes: List[pm.Note]) -> Tuple[float, float, int]:
    """retorna (poly_mean, poly_p95, poly_max) via varredura de eventos."""
    if not notes:
        return 0.0, 0.0, 0
    events = []
    for n in notes:
        events.append((n.start, +1))
        events.append((n.end,   -1))
    events.sort(key=lambda x: (x[0], -x[1]))
    active = 0
    last_t = events[0][0]
    end_t = max((n.end for n in notes), default=last_t)
    # amostra por segmentos: [(dur, active)]
    seg_vals = []
    poly_max = 0
    for t, d in events:
        if t > last_t:
            seg_vals.append((t - last_t, active))
            last_t = t
        active += d
        if active > poly_max:
            poly_max = active
    dur_total = max(1e-9, end_t - events[0][0])
    # média ponderada
    mean = sum(dur*v for dur, v in seg_vals) / dur_total if seg_vals else 0.0
    # p95 ponderado (quantil com pesos)
    if seg_vals:
        vals = np.array([v for _, v in seg_vals], dtype=float)
        wts  = np.array([dur for dur, _ in seg_vals], dtype=float)
        order = np.argsort(vals)
        cw = np.cumsum(wts[order] / (wts.sum() + 1e-12))
        p95 = float(vals[order][np.searchsorted(cw, 0.95)])
    else:
        p95 = 0.0
    return float(mean), float(p95), int(poly_max)

def _program_palette(pmid: pm.PrettyMIDI) -> Dict[str, Any]:
    progs = sorted({int(inst.program) for inst in pmid.instruments if not inst.is_drum})
    has_drums = any(inst.is_drum for inst in pmid.instruments)
    return {"programs": progs, "has_drums": has_drums}

def _tempo_profile(src: Path) -> List[Dict[str, float]]:
    # lista (t acumulado em s, bpm atual) de cada set_tempo; se não houver, retorna [].
    try:
        import mido
        mid = mido.MidiFile(src)
        tpb = mid.ticks_per_beat
        uspb = 500_000  # 120 bpm default
        t_acc = 0.0
        profile = [{"t": 0.0, "bpm": 60_000_000 / uspb}]
        for msg in mido.merge_tracks(mid.tracks):
            if msg.time:
                t_acc += mido.tick2second(msg.time, tpb, uspb)
            if msg.is_meta and msg.type == "set_tempo":
                uspb = max(1, int(msg.tempo))
                profile.append({"t": float(t_acc), "bpm": float(60_000_000 / uspb)})
        # se houver só o default, e nenhum set_tempo, devolve lista vazia (mapa único)
        return profile if len(profile) > 1 else []
    except Exception:
        return []

@st.cache_data(show_spinner=False)
def _analyze_cached(path_str: str) -> Dict[str, Any]:
    """wrap p/ cache: roda anal13_adapter + derivados para UI TG."""
    src = Path(path_str)
    out_dir = src.parent / "_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"{src.stem}_metrics.csv"

    # adapter 13 (gera CSV e devolve dataframe + global/tracks)
    res = analyze_v13(source_midi=src, out_dir=out_dir, out_name=out_name)

    # PrettyMIDI para extras visuais
    pmid = pm.PrettyMIDI(str(src))
    bars = _bar_boundaries(pmid)
    all_notes = [n for inst in pmid.instruments for n in inst.notes]

    poly_mean, poly_p95, poly_max = _polyphony_profile(all_notes)

    return {
        "file": str(src),
        "csv_path": str(res["csv_path"]),
        "df": res["dataframe"],  # pode ser usado em uma expander
        # global do adapter:
        "duration_seconds": float(res["global"].get("duration_seconds", pmid.get_end_time())),
        "time_signature_str": res["global"].get("time_signature", "4/4"),
        "tempo_bpm_min": float(res["global"].get("bpm_min", 0.0)),
        "tempo_bpm_median": float(res["global"].get("bpm_median", 0.0)),
        "tempo_bpm_max": float(res["global"].get("bpm_max", 0.0)),
        "bpm_source": res["global"].get("bpm_source", "—"),
        # TG extras:
        "duration_hhmmss": _human_hhmmss(float(res["global"].get("duration_seconds", 0.0))),
        "bars_estimated": max(0, len(bars) - 1),
        "time_signature": {
            "numerator": int(res["global"].get("time_signature", "4/4").split("/")[0]),
            "denominator": int(res["global"].get("time_signature", "4/4").split("/")[1]),
        },
        "polyphony": {"mean": poly_mean, "p95": poly_p95, "max": poly_max},
        "program_palette": _program_palette(pmid),
        "pitch_class_histogram": _pitch_class_histogram(pmid),
        "tempo_profile": _tempo_profile(src),
    }

def _csv_bytes_from_df(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False, float_format="%.6f")
    return buf.getvalue().encode("utf-8")

def _sibling_wav_for(midi_path: Path) -> Path | None:
    cand = midi_path.with_suffix(".wav")
    if cand.exists(): return cand
    # tenta procurar na mesma pasta por nome-base
    for p in midi_path.parent.glob(f"{midi_path.stem}*.wav"):
        return p
    return None

# ======================== Paths do Projeto ========================
ROOT = Path(__file__).resolve().parents[1]   # /midicaps
SRC  = ROOT / "src"                          # /midicaps/src
DATA = ROOT / "data"                         # /midicaps/data

OUT       = DATA / "infer_out"               # base das saídas
SESSIONS  = OUT                              # onde criaremos run_YYYYMMDD_HHMMSS
CKPTS     = ROOT / "runs"                    # checkpoints de modelo (softprompt_epXX.pt)
BIN       = DATA / "binpack"
CTRL      = DATA / "ctrl_vocab.json"
SF2       = ROOT / "assets" / "TimGM6mb.sf2"  # opcional (para render WAV no adapter)

OUT.mkdir(parents=True, exist_ok=True)

# ======================== Imports de Adapters ========================
from adapters.gen10_adapter import generate_v10
from adapters.cont11_adapter import continue_v11
from adapters.edit12_adapter import edit_v12
from adapters.anal13_adapter import analyze_v13


from adapters.run_utils import (
    start_new_run, list_runs, next_seq_id, seq_paths, mark_old
)

# ======================== Config da Página ========================
st.set_page_config(page_title="TG – Música Procedural", layout="wide")


# ======================== Estado / Utils ========================
def _init_session():
    defaults = {
        # players por aba (mantidos para outras abas/uso futuro)
        "gen_prev_audio": None, "gen_new_audio": None,
        "cont_prev_audio": None, "cont_new_audio": None,
        "edit_prev_audio": None, "edit_new_audio": None,

        # últimos parâmetros
        "gen_last_params": None,
        "cont_last_params": None,
        "edit_last_params": None,

        # run atual e último id
        "run_dir": None,            # Path
        "last_seq_id": None,        # str

        # paths atuais
        "current_midi_path": None,
        "current_wav_path": None,

        # nome do projeto
        "current_name": "untitled",

        # histórico simples (labels)
        "versions": [],
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def _load_bytes_safe(p: Path) -> bytes | None:
    try:
        return p.read_bytes() if p and p.exists() else None
    except Exception:
        return None


def _after_success_generation(tab_key_prefix: str, wav_path: Path, midi_path: Path, seq_id: str):
    """
    Atualiza players da aba: move 'novo' -> 'antes', carrega 'novo' do disco.
    tab_key_prefix ∈ {'gen','cont','edit'}
    """
    prev_key = f"{tab_key_prefix}_prev_audio"
    new_key  = f"{tab_key_prefix}_new_audio"

    # move o 'novo' anterior para 'antes' (mantido p/ compat.)
    st.session_state[prev_key] = st.session_state.get(new_key)

    # carrega o wav recém-gerado
    wav_bytes = _load_bytes_safe(wav_path)
    st.session_state[new_key] = wav_bytes

    # guarda infos úteis
    st.session_state["last_seq_id"] = seq_id
    st.session_state["current_midi_path"] = str(midi_path) if midi_path else None
    st.session_state["current_wav_path"]  = str(wav_path) if wav_path.exists() else None


def _list_run_wavs(run_dir: Path) -> list[Path]:
    """Lista todos os WAVs da run atual, ordenados por seq (0001.wav, 0002.wav, ...)."""
    if not run_dir or not Path(run_dir).exists():
        return []
    return sorted(Path(run_dir).glob("*.wav"), key=lambda p: p.name)


_init_session()

# ======================== Cabeçalho / Barra de Runs ========================
st.markdown(
    "<h2 style='margin-top:0'>🎵 TG – Geração Procedural de Trilhas</h2>"
    "<p style='opacity:0.8'>Geração • Continuação • Edição de Seções</p>",
    unsafe_allow_html=True,
)

bar = st.container()
colA, colB = bar.columns([1, 2])
with colA:
    if st.button("🆕 Nova Run"):
        st.session_state.run_dir = start_new_run(SESSIONS)
        st.session_state.last_seq_id = None
        st.toast(f"Nova run: {Path(st.session_state.run_dir).name}")

with colB:
    runs = list_runs(SESSIONS)
    if runs:
        pick = st.selectbox("Selecionar run existente", [r.name for r in runs], index=len(runs)-1)
        if pick:
            st.session_state.run_dir = next(r for r in runs if r.name == pick)
    else:
        st.info("Nenhuma run ainda. Clique em '🆕 Nova Run'.")

st.caption(f"Run atual: {st.session_state.run_dir if st.session_state.run_dir else '—'}")

# ======================== Abas ========================
tab_gen, tab_cont, tab_edit, tab_analysis = st.tabs(["Geração", "Continuar", "Editar trecho", "Análise"])

# ======================== Aba 1: Geração (10) ========================
with tab_gen:
    left, main, _sp = st.columns([1.1, 2.2, 0.2])

    # Painel de parâmetros
    with left:
        st.markdown("### Parâmetros de Geração")
        with st.form("form_gen"):

            # texto-condição
            prompt = st.text_input("🎨 Prompt textual", "dark epic orchestral with choir and low strings")

            # tempo e compasso
            bpm = st.number_input("🎵 Tempo (BPM)", min_value=40, max_value=220, value=120, step=1)
            ts = st.selectbox("🕓 Compasso", ["4/4", "3/4", "6/8"], index=0)

            # duração e estrutura
            primer_bars = st.number_input("🔹 Compassos de primer (trecho inicial neutro)", 
                                        min_value=0, max_value=64, value=8, step=1)
            # dur_bars = st.number_input("⏱️ Duração total (barras)", 
                                    # min_value=4, max_value=256, value=16, step=1)
            duration_seconds = st.number_input("⏳ Duração alvo (segundos, aproximado)", 
                                            min_value=5, max_value=600, value=30, step=5)

            # sampling
            temperature = st.slider("🌡️ Temperatura (aleatoriedade)", 0.1, 2.0, 1.2, 0.05)
            top_p = st.slider("🎯 Top-p (nucleus sampling)", 0.1, 1.0, 0.95, 0.01)
            # max_new_tokens = st.number_input("🧮 Máx. novos tokens gerados", 
            #                                 min_value=100, max_value=5000, value=3000, step=100)

            # controle de instrumentos
            allow_programs = st.text_input("🎻 Programas permitidos (ex.: 40,41,42...)", 
                                        "40,41,42,43,44,45,46,47,48")
            ban_drums = st.checkbox("🚫 Banir bateria (drums)", value=True)

            # # checkpoint e dataset (opcional)
            # soft_ckpt = st.text_input("🧩 Checkpoint do modelo (.pt)", 
            #                         str(CKPTS / "softprompt_midicaps" / "softprompt_ep02.pt"))
            # captions_csv = st.text_input("📄 CSV de legendas (captions)", 
            #                             str(DATA / "splits" / "train_5k.csv"))
            # tokens_jsonl = st.text_input("💾 Tokens JSONL", 
            #                             str(DATA / "tokens" / "train_5k.jsonl"))

            submitted = st.form_submit_button("🎶 Gerar Música")


        if submitted:
            if not st.session_state.run_dir:
                st.warning("Crie/seleciona uma run primeiro (botão 🆕 Nova Run).")
            else:
                # guarda params (referência futura)
                st.session_state.gen_last_params = dict(
                    prompt=prompt, bpm=bpm, ts=ts,
                    allow_programs=allow_programs, ban_drums=ban_drums
                )

                # próximo ID sequencial
                seq_id = next_seq_id(Path(st.session_state.run_dir))

                resp = generate_v10(
                    repo_root=ROOT,
                    bin_dir=BIN,
                    soft_ckpt=CKPTS / "softprompt_midicaps" / "softprompt_ep02.pt",
                    ctrl_vocab=CTRL,
                    out_dir=OUT,
                    script10=SRC / "10_infer_text_softprompt.py",
                    run_dir=Path(st.session_state.run_dir),
                    seq_id=seq_id,
                    prompt_text=prompt,
                    bpm=bpm,
                    time_signature=ts,
                    duration_seconds=duration_seconds,
                    allow_programs=allow_programs,
                    ban_drums=ban_drums,
                    soundfont=(SF2 if SF2.exists() else None),
                    # max_new_tokens=6000,
                )

                if resp["ok"]:
                    midi_path, wav_path = seq_paths(Path(st.session_state.run_dir), seq_id)
                    if not wav_path.exists():
                        st.warning("WAV não foi gerado (SoundFont ausente?).")
                    _after_success_generation("gen", wav_path, midi_path, seq_id)
                    st.success(f"Geração ok. MIDI: {midi_path} | WAV: {wav_path if wav_path.exists() else '—'}")
                else:
                    st.error("Falha na geração.")
                    st.code(resp["stderr"] or resp["stdout"] or "sem logs", language="bash")

    # Lista todas as faixas geradas na run atual (Wavesurfer.js)
    with main:
        st.markdown("### Músicas geradas nesta run")
        if not st.session_state.run_dir:
            st.info("Nenhuma run selecionada. Clique em 🆕 Nova Run.")
        else:
            wavs = _list_run_wavs(Path(st.session_state.run_dir))
            if not wavs:
                st.info("Nenhum WAV nesta run ainda. Gere uma música para começar.")
            else:
                for wav_path in reversed(wavs):
                    wavesurfer_player(Path(wav_path), title=f"File {wav_path.name}")
                        # wavesurfer(
                        #     # key=f"ws-{seq}",
                        #     audio_src=wav_path,
                        #     # plugins=["regions", "timeline", "zoom"],
                        #     # wave_options=WaveSurferOptions(height=10,),
                        # )
                    # st.divider()

# ======================== Aba 2: Continuar (11) ========================
with tab_cont:
    left, main, _sp = st.columns([1.1, 2.2, 0.2])

    # ---------- Painel esquerdo: parâmetros ----------
    with left:
        

        # Seletor de arquivo MIDI (base para continuar)
        midi_files = []
        if st.session_state.run_dir and Path(st.session_state.run_dir).exists():
            midi_files = sorted(Path(st.session_state.run_dir).glob("*.mid"))

        def _on_midi_change():
            # quando trocar o MIDI base, limpamos os previews de continuação
            st.session_state.current_wav_path = None

        midi_choice = st.selectbox(
            "🎼 Escolher MIDI base",
            [p.name for p in midi_files] if midi_files else ["(nenhum disponível)"],
            key="cont_midi_choice",
            on_change=_on_midi_change,
        )
        midi_path = (Path(st.session_state.run_dir) / midi_choice) if midi_files else None

        with st.form("form_cont"):
            st.markdown("### Parâmetros de Continuação")
            prompt_text = st.text_input("📝 Prompt de guia (vazio = só primer + controles)", "")
            bars_to_use = st.number_input("📏 Barras finais usadas como primer",
                                        min_value=1, max_value=32, value=8, step=1)
            duration_seconds = st.number_input("⏳ Duração a gerar (segundos)",
                                            min_value=5, max_value=600, value=30, step=5)
            tempo_bpm = st.number_input("🎵 BPM (tempo)", min_value=40, max_value=220, value=120)
            ts = st.selectbox("🕓 Compasso", ["4/4", "3/4", "6/8"], index=0)

            temperature = st.slider("🌡️ Temperatura (aleatoriedade)", 0.1, 2.0, 1.1, 0.05)
            top_p = st.slider("🎯 Top-p (nucleus sampling)", 0.1, 1.0, 0.95, 0.01)
            top_k = st.number_input("🔢 Top-k (0 desativa)", min_value=0, max_value=256, value=0, step=1)

            allow_programs = st.text_input("🎻 Programas permitidos (ex.: 40,41,42...)", "40,41,42,43,44,45,46,47,48")
            ban_drums = st.checkbox("🚫 Banir bateria (drums)", value=True)

            # st.markdown("### Penalizações (opcional)")
            # rest_penalty = st.slider("🛑 Penalidade de rests (0 = off)", 0.0, 2.0, 0.0, 0.05)
            # timeshift_penalty = st.slider("⏱️ Penalidade de timeshifts (0 = off)", 0.0, 2.0, 0.0, 0.05)



            # checkpoints (11 não usa captions/tokens_jsonl)
            soft_ckpt = str(CKPTS / "softprompt_midicaps" / "softprompt_ep02.pt")

            submitted = st.form_submit_button("🎶 Gerar Continuação")

        # ---------- Execução ----------
            if submitted:
                if not midi_path or not midi_path.exists():
                    st.warning("Nenhum arquivo MIDI selecionado.")
                elif not st.session_state.run_dir:
                    st.warning("Crie ou selecione uma run primeiro.")
                else:
                    # NÃO gere seq_id aqui; o adaptador já cuida disso
                    res = continue_v11(
                        repo_root=ROOT,
                        run_dir=Path(st.session_state.run_dir),
                        bin_dir=BIN,
                        ctrl_vocab=CTRL,
                        out_dir=OUT,  # use OUT (base), o adaptador move p/ run_dir
                        source_midi=midi_path,
                        soft_ckpt=Path(soft_ckpt),
                        prompt_text=prompt_text,
                        tempo_bpm=tempo_bpm,
                        time_signature=ts,
                        duration_seconds=duration_seconds,
                        tail_bars=bars_to_use,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        # max_new_tokens=max_new_tokens,  # descomente se expuser no form
                        allow_programs=allow_programs,
                        ban_drums=ban_drums,
                        soundfont=SF2,
                        # rest_penalty=rest_penalty,          # se reativar os sliders
                        # timeshift_penalty=timeshift_penalty,
                    )

                    if res["ok"]:
                        midi_out = Path(res["midi_path"]) if res["midi_path"] else None
                        wav_full = Path(res["wav_path"]) if res["wav_path"] else None
                        added_wav = Path(res["added_wav_path"]) if res.get("added_wav_path") else None

                        # atualiza estado (usa o seq_id calculado no adaptador)
                        seq_id = res["meta"]["seq_id"]
                        _after_success_generation("cont", wav_full, midi_out, seq_id)

                        st.success("Continuação gerada com sucesso!")
                    else:
                        st.error("Falha na continuação.")
                        st.code(res["stderr"] or res["stdout"] or "(sem saída)", language="bash")

    # ---------- Painel direito: players ----------
    with main:
        st.markdown("### 🎧 Comparação da Continuação")

        if not st.session_state.run_dir:
            st.info("Nenhuma run selecionada.")
        elif not midi_files or not midi_choice.endswith(".mid"):
            st.info("Nenhum MIDI disponível na run atual.")
        else:
            base_wav = Path(st.session_state.run_dir) / midi_choice.replace(".mid", ".wav")
            if base_wav.is_file():
                wavesurfer_player(base_wav, title="🎵 Original")

            new_wav_path = st.session_state.get("current_wav_path")
            if new_wav_path:
                new_wav = Path(new_wav_path)
                if new_wav.is_file():
                    wavesurfer_player(new_wav, title="🎼 Com adição (Final)")
                    added_wav = new_wav.with_name(f"{new_wav.stem}_added.wav")
                    if added_wav.is_file():
                        wavesurfer_player(added_wav, title="🎶 Somente trecho adicionado")


# ======================== Aba 3: Editar trecho (12) ========================
with tab_edit:
    st.markdown("### ✂️ Edição modular (layout vertical)")
    soft_ckpt = str(CKPTS / "softprompt_midicaps" / "softprompt_ep02.pt")

    # -------------------- 1) Seleção de MIDI (FORA DO FORM) --------------------
    midi_files = []
    if st.session_state.run_dir and Path(st.session_state.run_dir).exists():
        midi_files = sorted(Path(st.session_state.run_dir).glob("*.mid"))

    if "edits_by_midi" not in st.session_state:
        st.session_state.edits_by_midi = {}

    def _on_edit_midi_change():
        st.session_state.current_edit_wav = None

    # Linha visual em 3 colunas para o seletor de MIDI (fora do form)
    c1, c2, c3 = st.columns(3)
    with c1:
        midi_choice = st.selectbox(
            "🎼 Arquivo MIDI",
            [p.name for p in midi_files] if midi_files else ["(nenhum disponível)"],
            key="edit_midi_choice",
            on_change=_on_edit_midi_change,
        )
    with c2:
        st.empty()
    with c3:
        st.empty()

    source_midi = (Path(st.session_state.run_dir) / midi_choice) if midi_files else None
    source_wav  = source_midi.with_suffix(".wav") if source_midi else None
    midi_key    = str(source_midi.resolve()) if source_midi else None

    # -------------------- 2) FORM — Inputs (3 colunas) + seletor + botão --------------------
    # -------------------- 2A) Seletor de tempo (FORA DO FORM) --------------------
    st.markdown("#### Seletor de tempo")

    skey = f"edit_sel::{source_wav.resolve()}" if (source_wav and source_wav.exists()) else None
    last = st.session_state.get(skey, {"start": 0.0, "end": 5.0})

    # bootstrap
    if skey and skey not in st.session_state:
        st.session_state[skey] = {"start": float(last["start"]), "end": float(last["end"])}

    val = None
    if source_wav and source_wav.exists():
        current = st.session_state.get(skey, last)
        val = wavesurfer_region_selector(
            wav_path=source_wav,
            init_start=float(current["start"]),
            init_end=float(current["end"]),
            key=f"ws::{source_wav.resolve()}",   # CHAVE ESTÁVEL
        )

    # aplica retorno se houver
    if isinstance(val, dict) and "start" in val and "end" in val:
        st.session_state[skey] = {"start": float(val["start"]), "end": float(val["end"])}
        # if val.get("forceRerun"):
        #     st.rerun()  # só quando clicar "Usar esta seleção"

    # UI segura
    sel = st.session_state.get(skey, last)
    st.caption(f"Seleção: {sel['start']:.3f}s → {sel['end']:.3f}s")





    # -------------------- 2B) FORM — Inputs (3 colunas) + botão --------------------
    with st.form("form_edit_vertical", clear_on_submit=False):
        st.subheader("Parâmetros")

        col1, col2, col3 = st.columns(3)
        with col1:
            prompt_text = st.text_input("📝 Prompt de guia (vazio = só primer + controles)", "")
            tempo_bpm   = st.number_input("🎵 BPM (tempo)", min_value=40, max_value=220, value=120)
            context_bars_before = st.number_input("Context bars BEFORE", min_value=0, max_value=220, value=4)

        with col2:
            ts          = st.selectbox("🕓 Compasso", ["4/4", "3/4", "6/8"], index=0)
            temperature = st.slider("🌡️ Temperatura (aleatoriedade)", 0.1, 2.0, 1.1, 0.05)
            context_bars_after  = st.number_input("Context bars AFTER",  min_value=0, max_value=220, value=4)

        with col3:
            top_p = st.slider("🎯 Top-p (nucleus)", 0.1, 1.0, 0.95, 0.01)
            top_k = st.number_input("🔢 Top-k (0 desativa)", min_value=0, max_value=256, value=0, step=1)
            allow_programs = st.text_input("🎻 Programas permitidos (ex.: 40,41,42...)", "40,41,42,43,44,45,46,47,48")
            ban_drums      = st.checkbox("🚫 Banir bateria (drums)", value=True)

        submitted_edit = st.form_submit_button("GERAR", use_container_width=True)

    # -------------------- 3) Execução --------------------
    if submitted_edit:
        cur_sel = st.session_state.get(skey) if skey else None
        if not isinstance(cur_sel, dict):
            st.warning("Selecione um trecho no waveform antes de gerar (arraste/crie a região).")
            st.stop()

        try:
            t0_s = float(cur_sel["start"])
            t1_s = float(cur_sel["end"])
        except Exception:
            st.error("Falha ao ler a seleção de tempo.")
            st.stop()

        if t1_s <= t0_s + 1e-3:
            st.warning("Seleção inválida (end ≤ start). Ajuste a região e tente novamente.")
            st.stop()

        with st.spinner("Gerando edição..."):
            res = edit_v12(
                repo_root=ROOT,
                out_dir=OUT,
                run_dir=Path(st.session_state.run_dir),
                source_midi=source_midi,
                start_seconds=t0_s,
                end_seconds=t1_s,
                context_bars_before=context_bars_before,
                context_bars_after=context_bars_after,
                bin_dir=BIN,
                soft_ckpt=Path(soft_ckpt),
                ctrl_vocab=CTRL,
                prompt_text=prompt_text,
                tempo_bpm=tempo_bpm,
                time_signature=ts,
                temperature=temperature,
                top_p=top_p,
                allow_programs=allow_programs,
                ban_drums=ban_drums,
                soundfont=SF2,
            )
            
            if res.get("ok"):
                wav_path = Path(res.get("wav_path",""))
                if midi_key and wav_path.exists():
                    st.session_state.edits_by_midi[midi_key] = str(wav_path)
                    st.session_state.current_edit_wav = str(wav_path)
                st.success("Edição gerada com sucesso!")
            else:
                st.error("Falha na edição.")
                st.code(res.get("stderr") or res.get("stdout") or "(sem saída)", language="bash")


    # -------------------- 4) Players (vertical) --------------------
    st.divider()
    st.markdown("### 🎵 Música original")
    if source_wav and source_wav.exists():
        wavesurfer_player(source_wav, title="Original")
    else:
        st.info("Carregue um MIDI válido para visualizar o áudio original.")

    st.markdown("### 🎼 Música editada")
    edited_wav_path = st.session_state.edits_by_midi.get(midi_key) if midi_key else None
    if edited_wav_path and Path(edited_wav_path).exists():
        wavesurfer_player(Path(edited_wav_path), title="Editado")
    else:
        st.info("Gere uma edição para visualizar aqui.")




# ======================== Aba 4: Análise ========================
with tab_analysis:
    left, right = st.columns([1.1, 2.2])

    # ======================================================
    # ESQUERDA — Seletor de Música
    # ======================================================
    with left:
        st.markdown("### 🎼 Selecionar Música para Análise")

        origem = st.radio("Origem", ["Selecionar existente", "Upload"], horizontal=True)

        midi_path = None
        base_dir = Path(st.session_state.get("run_dir") or ".")

        if origem == "Selecionar existente":
            midi_candidates = sorted([p for p in base_dir.rglob("*.mid")] + [p for p in base_dir.rglob("*.midi")])
            if not midi_candidates:
                st.info("Nenhum MIDI encontrado na run atual.")
            else:
                pick = st.selectbox(
                    "Escolha o MIDI",
                    midi_candidates,
                    index=0,
                    format_func=lambda p: str(p.relative_to(base_dir)) if str(p).startswith(str(base_dir)) else str(p)
                )
                if st.button("📊 Analisar selecionado", use_container_width=True):
                    st.session_state.current_midi_path = str(Path(pick).resolve())
                    st.toast(f"Carregado: {Path(pick).name}")
        else:
            up = st.file_uploader("Selecione um arquivo .mid/.midi", type=["mid", "midi"])
            if up is not None:
                session_dir = Path(st.session_state.get("run_dir") or tempfile.gettempdir())
                session_dir.mkdir(parents=True, exist_ok=True)
                dst = session_dir / f"uploaded_{up.name}"
                with open(dst, "wb") as f:
                    f.write(up.read())
                st.session_state.current_midi_path = str(dst.resolve())
                st.toast(f"MIDI carregado: {dst.name}")

    # ======================================================
    # DIREITA — Player + Métricas
    # ======================================================
    with right:
        st.markdown("### 🔍 Análise Musical")

        midi_path_str = st.session_state.get("current_midi_path")
        if not midi_path_str:
            st.info("Selecione um MIDI à esquerda para iniciar a análise.")
            st.stop()

        midi_path = Path(midi_path_str)
        if not midi_path.exists():
            st.error(f"O arquivo não existe mais: {midi_path}")
            st.stop()

        with st.spinner("Calculando métricas..."):
            try:
                metrics = _analyze_cached(str(midi_path))
                st.session_state.last_metrics = metrics
            except Exception as e:
                st.error(f"Falha ao analisar MIDI: {e}")
                st.stop()

        # ---------- PLAYER ----------
        wav_path = _sibling_wav_for(midi_path)
        st.markdown("### 🎧 Player de Áudio")
        if wav_path and wav_path.exists():
            with open(wav_path, "rb") as f:
                st.audio(f.read(), format="audio/wav")
            st.caption(f"WAV encontrado: {wav_path.name}")
        else:
            st.info("Nenhum WAV irmão encontrado.")

        # ---------- MÉTRICAS PRINCIPAIS (TG) ----------
        st.markdown("### 📈 Métricas do TG")

        # bloco 1 — parâmetros gerais
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Duração", metrics["duration_hhmmss"])
        c2.metric("Tempo (BPM mediana)", f'{metrics["tempo_bpm_median"]:.1f}' if metrics["tempo_bpm_median"] else "—")
        ts = metrics["time_signature"]
        c3.metric("Compasso", f'{ts["numerator"]}/{ts["denominator"]}')
        c4.metric("Barras estimadas", metrics["bars_estimated"])

        # bloco 2 — polifonia
        poly = metrics["polyphony"]
        c5, c6, c7 = st.columns(3)
        c5.metric("Polifonia média", f'{poly["mean"]:.2f}')
        c6.metric("Polifonia p95", f'{poly["p95"]:.0f}')
        c7.metric("Máx. polifonia", f'{poly["max"]}')

        # ---------- MÉTRICAS AVANÇADAS (variedade, coerência, loopabilidade) ----------
        st.markdown("### 🎵 Métricas Musicais Derivadas")

        # tenta buscar do CSV completo se já estiver calculado
        df_metrics = metrics.get("df")
        loop_score = None
        melodic_var = None
        temporal_coh = None

        # tenta ler se as colunas existem (funções adicionadas no 13)
        if isinstance(df_metrics, pd.DataFrame) and "melodic_variety" in df_metrics.columns:
            melodic_var = df_metrics.loc[df_metrics["scope"]=="GLOBAL","melodic_variety"].values[0]
        if isinstance(df_metrics, pd.DataFrame) and "temporal_coherence" in df_metrics.columns:
            temporal_coh = df_metrics.loc[df_metrics["scope"]=="GLOBAL","temporal_coherence"].values[0]
        if isinstance(df_metrics, pd.DataFrame) and "loopability" in df_metrics.columns:
            loop_score = df_metrics.loc[df_metrics["scope"]=="GLOBAL","loopability"].values[0]

        c9, c10, c11, c12 = st.columns(4)
        c9.metric("Variedade Melódica", f'{melodic_var:.2f}' if melodic_var else "—")
        c10.metric("Coerência Temporal", f'{temporal_coh:.2f}' if temporal_coh else "—")
        c11.metric("Loopabilidade", f'{loop_score:.2f}' if loop_score else "—")
        c12.metric("Fidelidade BPM", f'{metrics["tempo_bpm_median"]:.1f}' if metrics["tempo_bpm_median"] else "—")

        # ---------- GRÁFICOS ----------
        with st.expander("Histograma de Classe de Altura (C–B)"):
            labels = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
            df_pc = pd.DataFrame({"PitchClass": labels, "Proporção": metrics["pitch_class_histogram"]})
            st.bar_chart(df_pc.set_index("PitchClass"))

        with st.expander("Mapa de Mudanças de Tempo (BPM)"):
            if metrics["tempo_profile"]:
                st.dataframe(
                    [{"t (s)": f'{row["t"]:.2f}', "BPM": f'{row["bpm"]:.2f}'} for row in metrics["tempo_profile"]],
                    use_container_width=True
                )
            else:
                st.caption("Mapa de tempo único (sem variações).")

        with st.expander("Tabela completa (CSV)"):
            st.dataframe(metrics["df"], use_container_width=True, hide_index=True)
