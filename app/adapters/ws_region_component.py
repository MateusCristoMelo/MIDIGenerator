# ws_region_component.py
from __future__ import annotations
from pathlib import Path
import base64
import streamlit as st
import streamlit.components.v1 as components

# Em DEV: aponta para o servidor do frontend (vite)
# Em PROD: troque por path=... (pasta build exportada)
_ws_region = components.declare_component(
    "ws_region",
    url="http://localhost:3001"   # <- ajuste porta conforme seu npm run start
    # path=".../frontend/build"   # <- use isso em produção
)

def wavesurfer_region(
    wav_path: Path,
    *,
    init_start: float = 0.0,
    init_end: float = 5.0,
    height: int = 220,
    key: str | None = None,
) -> dict | None:
    """
    Componente bi-direcional (oficial) Streamlit:
    - Mostra waveform com Wavesurfer + Regions
    - Retorna {start, end} via Streamlit.setComponentValue
    """
    p = Path(wav_path)
    if not p.exists():
        st.warning(f"Áudio não encontrado: {wav_path}")
        return None

    # Embeda WAV para evitar CORS
    b64 = base64.b64encode(p.read_bytes()).decode()
    data_url = f"data:audio/wav;base64,{b64}"

    # Envia argumentos (qualquer JSON-serializável)
    value = _ws_region(
        audio_src=data_url,
        init_start=float(init_start),
        init_end=float(init_end),
        desired_height=int(height),
        key=key,
        default=None,  # valor inicial de retorno (antes do primeiro setComponentValue)
    )

    # value é o que o frontend enviou via Streamlit.setComponentValue(...)
    # esperamos um dict {"start": float, "end": float}
    if isinstance(value, dict) and "start" in value and "end" in value:
        try:
            return {"start": float(value["start"]), "end": float(value["end"])}
        except Exception:
            return None
    return None