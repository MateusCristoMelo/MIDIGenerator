import "./styles.css";
import WaveSurfer from "wavesurfer.js";
import RegionsPlugin from "wavesurfer.js/dist/plugins/regions.esm.js";
import { Streamlit, RenderData } from "streamlit-component-lib";

let ws: WaveSurfer | null = null;
let regions: any = null;
let active: any = null;

let connected = false;           // só envia msgs ao host quando true
let lastSig = "";                // evita remontar com os mesmos args
let mountSeq = 0;                // invalida handlers de montagens antigas

function fmt(t: number) {
  const s = Math.max(0, Math.floor(t || 0));
  const m = Math.floor(s / 60);
  const r = s % 60;
  return `${m}:${String(r).padStart(2, "0")}`;
}

function safeSetHeight(h: number) {
  if (connected) Streamlit.setFrameHeight(Math.max(220, Math.ceil(h) + 8));
}

function mount(renderData: RenderData) {
  const args = renderData.args || {};
  const audioUrl: string = args.audioUrl;
  const initStart: number = args.initStart ?? 0.0;
  const initEnd: number = args.initEnd ?? 5.0;

  // assinatura dos args pra não remontar à toa
  const sig = JSON.stringify({
    len: typeof audioUrl === "string" ? audioUrl.length : 0,
    s: Math.round(initStart * 1000),
    e: Math.round(initEnd * 1000),
  });
  if (sig === lastSig) {
    // nada mudou: só ajusta altura e sai
    const root = document.getElementById("rootwrap");
    if (root) safeSetHeight(root.getBoundingClientRect().height);
    return;
  }
  lastSig = sig;

  // invalida montagens anteriores
  const mySeq = ++mountSeq;

  // destrói instância anterior com try/catch para evitar AbortError barulhento
  try { ws?.destroy(); } catch {}
  ws = null;
  active = null;

  // UI raiz observável
  document.body.innerHTML = `
    <div id="rootwrap" class="wrap">
      <div style="font-weight:600;margin-bottom:6px">Selecione um trecho (arraste no waveform ou ajuste as bordas)</div>
      <div id="wave"></div>
      <div class="row">
        <button id="play">▶️ Play/Pause</button>
        <button id="apply">✅ Usar esta seleção</button>
        <span id="lbl"></span>
      </div>
    </div>
  `;

  const root = document.getElementById("rootwrap")!;
  const container = document.getElementById("wave")!;
  const lbl = document.getElementById("lbl")!;
  const playBtn = document.getElementById("play")!;
  const applyBtn = document.getElementById("apply")!;

  // ResizeObserver: só envia após conectado
  let lastH = -1;
  const ro = new ResizeObserver(() => {
    const h = Math.ceil(root.getBoundingClientRect().height);
    if (h !== lastH) {
      lastH = h;
      safeSetHeight(h);
    }
  });
  ro.observe(root);

  regions = RegionsPlugin.create();
  try {
    ws = WaveSurfer.create({
      container,
      url: audioUrl,
      height: 96,
      waveColor: "#90a4ae",
      progressColor: "#26a69a",
      cursorColor: "#111",
      normalize: true,
      backend: "MediaElement",
      plugins: [regions],
    });
  } catch (e) {
    console.warn("WaveSurfer create error:", e);
    return;
  }

  ws.on("error", (e: any) => {
    // ignora AbortError silenciosamente (causado por teardown durante reload)
    if (String(e?.name || e).includes("Abort")) {
      console.debug("WaveSurfer load aborted (likely rerender).");
      return;
    }
    console.error("WaveSurfer error:", e);
  });

  ws.on("ready", () => {
    // se outra montagem aconteceu, aborta este handler
    if (mySeq !== mountSeq) return;

    const dur = ws!.getDuration();
    const s = Math.min(initStart, Math.max(0, dur - 0.001));
    const e = Math.min(initEnd, dur);
    active = regions.addRegion({
      start: s, end: e, color: "rgba(255,165,0,0.25)", drag: true, resize: true,
    });
    lbl.textContent = `Seleção: ${fmt(s)} → ${fmt(e)} (dur: ${fmt(Math.max(0, e - s))})`;
    safeSetHeight(root.getBoundingClientRect().height);
  });

  regions.enableDragSelection({ color: "rgba(255,0,0,0.12)" });

  regions.on("region-created", (reg: any) => {
    if (mySeq !== mountSeq) return;
    if (active && active !== reg) active.remove();
    active = reg;
    lbl.textContent = `Seleção: ${fmt(reg.start)} → ${fmt(reg.end)} (dur: ${fmt(Math.max(0, reg.end - reg.start))})`;
  });

  regions.on("region-updated", (reg: any) => {
    if (mySeq !== mountSeq) return;
    if (active === reg) {
      lbl.textContent = `Seleção: ${fmt(reg.start)} → ${fmt(reg.end)} (dur: ${fmt(Math.max(0, reg.end - reg.start))})`;
    }
  });

  playBtn.addEventListener("click", () => ws?.playPause());

  // Só envia valor quando o usuário confirma
  applyBtn.addEventListener("click", () => {
    if (mySeq !== mountSeq || !active) return;
    const start = Math.max(0, Math.round(active.start * 1000) / 1000);
    const end   = Math.max(start + 0.001, Math.round(active.end   * 1000) / 1000);
    if (connected) {
      Streamlit.setComponentValue({ start, end, forceRerun: true });
    }
  });
}

// Handshake: marca como conectado, então monta.
// NÃO chama setComponentValue aqui.
function onRender(event: Event) {
  connected = true;
  const data = (event as CustomEvent<RenderData>).detail;
  mount(data);
}

Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, onRender);
Streamlit.setComponentReady();
