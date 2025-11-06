from pathlib import Path
import base64
import streamlit as st
import streamlit.components.v1 as components

_DIST = Path(__file__).parent / "frontend" / "dist"
_MC = components.declare_component(
    "my_component",
    path=str(_DIST) if _DIST.exists() else "http://localhost:5173",
)

def wavesurfer_region_selector(wav_path: Path, *, init_start=0.0, init_end=5.0, key=None):
    p = Path(wav_path)
    if not p.exists():
        st.warning(f"Áudio não encontrado: {wav_path}")
        return None
    # data URL estável
    b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
    data_url = f"data:audio/wav;base64,{b64}"
    # ⚠️ sem 'default' aqui (se passar, Streamlit retorna imediatamente e dá rerun)
    return _MC(
        audioUrl=data_url,
        initStart=float(init_start),
        initEnd=float(init_end),
        key=key,  # ex.: key=f"ws::{p.resolve()}"
    )
