# 12_edit_sections.py — Edição cirúrgica com primer e costura (independente da etapa 10)

import argparse, os, sys, math, tempfile, json
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pretty_midi as pm
import mido

# =========================
# Utilidades de compasso/tempo
# =========================

def _first_ts(pmidi: pm.PrettyMIDI) -> Tuple[int, int]:
    if pmidi.time_signature_changes:
        ts = pmidi.time_signature_changes[0]
        return ts.numerator, ts.denominator
    return 4, 4

def _beats_per_bar(num: int, den: int) -> int:
    return int(round(num * (4.0 / den)))

def _bar_boundaries(pmidi: pm.PrettyMIDI) -> List[float]:
    num, den = _first_ts(pmidi)
    bpb = max(1, _beats_per_bar(num, den))
    beats = pmidi.get_beats()
    if len(beats) == 0:
        dur = pmidi.get_end_time()
        if dur <= 0: 
            return [0.0, 0.0]
        grid = np.arange(0.0, dur + 1e-6, 2.0).tolist()  # fallback: 2s por compasso
        if grid[-1] < dur: grid.append(dur)
        return grid
    bars = []
    for i in range(0, len(beats), bpb):
        bars.append(beats[i])
    end_t = pmidi.get_end_time()
    if not bars or bars[0] > 1e-6:
        bars = [0.0] + bars
    if bars[-1] < end_t:
        bars.append(end_t)
    return bars

def _clip_midi(pmidi: pm.PrettyMIDI, t0: float, t1: float) -> pm.PrettyMIDI:
    t0 = max(0.0, t0)
    t1 = max(t0, t1)
    out = pm.PrettyMIDI(resolution=pmidi.resolution)
    for inst in pmidi.instruments:
        new_inst = pm.Instrument(program=inst.program, is_drum=inst.is_drum, name=inst.name)
        for n in inst.notes:
            if n.end <= t0 or n.start >= t1:
                continue
            ns = max(n.start, t0) - t0
            ne = min(n.end, t1) - t0
            if ne > ns + 1e-6:
                new_inst.notes.append(pm.Note(n.velocity, n.pitch, ns, ne))
        for b in inst.pitch_bends:
            if t0 <= b.time < t1:
                new_inst.pitch_bends.append(pm.PitchBend(b.pitch, b.time - t0))
        for c in inst.control_changes:
            if t0 <= c.time < t1:
                new_inst.control_changes.append(pm.ControlChange(c.number, c.value, c.time - t0))
        if new_inst.notes or new_inst.control_changes or new_inst.pitch_bends:
            out.instruments.append(new_inst)
    return out

def _scale_velocities(pmidi: pm.PrettyMIDI, t0: float, t1: float, from_gain: float, to_gain: float):
    if t1 <= t0: return
    dur = max(1e-6, t1 - t0)
    for inst in pmidi.instruments:
        for n in inst.notes:
            c = np.clip((n.start - t0) / dur, 0.0, 1.0)
            g = (1.0 - c) * from_gain + c * to_gain
            n.velocity = int(np.clip(round(n.velocity * g), 1, 127))

def _write_midi(pmidi: pm.PrettyMIDI, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pmidi.write(str(path))

# =========================
# ------- BLOCO GPT / SOFT-PROMPT (copiado/adaptado da etapa 10)
# =========================

import torch, torch.nn as nn, torch.nn.functional as F
from sentence_transformers import SentenceTransformer

PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0,1,2,3

class GPTConfig:
    def __init__(self, vocab_size, seq_len, n_layers=6, n_heads=8, d_model=512, d_ff=2048, dropout=0.1):
        self.vocab_size=vocab_size; self.seq_len=seq_len
        self.n_layers=n_layers; self.n_heads=n_heads
        self.d_model=d_model; self.d_ff=d_ff; self.dropout=dropout

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.nh=cfg.n_heads; self.hd=cfg.d_model//cfg.n_heads
        self.qkv=nn.Linear(cfg.d_model, 3*cfg.d_model)
        self.proj=nn.Linear(cfg.d_model, cfg.d_model)
        self.attn_drop=nn.Dropout(cfg.dropout); self.resid_drop=nn.Dropout(cfg.dropout)
        self.register_buffer("mask", torch.tril(torch.ones(cfg.seq_len, cfg.seq_len)).unsqueeze(0).unsqueeze(0))
    def forward(self,x):
        B,T,C=x.shape
        qkv=self.qkv(x); q,k,v=qkv.split(C,dim=2)
        q=q.view(B,T,self.nh,self.hd).transpose(1,2)
        k=k.view(B,T,self.nh,self.hd).transpose(1,2)
        v=v.view(B,T,self.nh,self.hd).transpose(1,2)
        att=(q@k.transpose(-2,-1))/math.sqrt(self.hd)
        att=att.masked_fill(self.mask[:,:,:T,:T]==0,float('-inf'))
        att=F.softmax(att,dim=-1); att=self.attn_drop(att)
        y=att@v; y=y.transpose(1,2).contiguous().view(B,T,C)
        return self.resid_drop(self.proj(y))

class MLP(nn.Module):
    def __init__(self,cfg): 
        super().__init__()
        self.fc1=nn.Linear(cfg.d_model,4*cfg.d_model); self.fc2=nn.Linear(4*cfg.d_model,cfg.d_model)
        self.drop=nn.Dropout(cfg.dropout); self.act=nn.GELU()
    def forward(self,x): return self.drop(self.fc2(self.act(self.fc1(x))))

class Block(nn.Module):
    def __init__(self,cfg): 
        super().__init__()
        self.ln1=nn.LayerNorm(cfg.d_model); self.attn=CausalSelfAttention(cfg)
        self.ln2=nn.LayerNorm(cfg.d_model); self.mlp=MLP(cfg)
    def forward(self,x): x=x+self.attn(self.ln1(x)); x=x+self.mlp(self.ln2(x)); return x

class GPT(nn.Module):
    def __init__(self,cfg: GPTConfig):
        super().__init__()
        self.cfg=cfg
        self.tok_emb=nn.Embedding(cfg.vocab_size,cfg.d_model)
        self.pos_emb=nn.Embedding(cfg.seq_len,cfg.d_model)
        self.drop=nn.Dropout(cfg.dropout)
        self.blocks=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f=nn.LayerNorm(cfg.d_model)
        self.head=nn.Linear(cfg.d_model,cfg.vocab_size,bias=False)
        self.apply(self._init)
    def _init(self,m):
        if isinstance(m,nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m,nn.Embedding):
            nn.init.normal_(m.weight,0.0,0.02)
    def forward_with_prefix(self, idx, prefix_emb=None):
        # idx: [B,T]; prefix_emb: [B,P,d_model] ou None
        B,T=idx.shape
        if prefix_emb is not None:
            P=prefix_emb.size(1); total=P+T
            assert total<=self.cfg.seq_len, f"comprimento {total}>{self.cfg.seq_len}"
            pos=torch.arange(total,device=idx.device).unsqueeze(0)
            x_tok=self.tok_emb(idx)
            x=torch.cat([prefix_emb,x_tok],dim=1)+self.pos_emb(pos)
        else:
            pos=torch.arange(T,device=idx.device).unsqueeze(0)
            x=self.tok_emb(idx)+self.pos_emb(pos)
        x=self.drop(x)
        for blk in self.blocks: x=blk(x)
        return self.head(self.ln_f(x))

class SoftPromptMapper(nn.Module):
    def __init__(self, dim_text, d_model, n_soft):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim_text, 2*d_model),
            nn.GELU(),
            nn.Linear(2*d_model, n_soft*d_model)
        )
        self.n_soft=n_soft; self.d_model=d_model
    def forward(self, emb):  # emb: [B, dim_text]
        x=self.proj(emb); return x.view(-1,self.n_soft,self.d_model)

# -------- util da 10 --------
def load_vocab(bin_dir: Path):
    stoi=json.loads((bin_dir/"vocab.json").read_text(encoding="utf-8"))
    itos={int(v):k for k,v in stoi.items()}
    return stoi, itos

def softmax_sample(logits, temperature=1.0, top_p=0.9):
    logits=logits.float()
    if temperature>0: logits=logits/temperature
    probs=logits.softmax(-1)
    sp, si=torch.sort(probs, descending=True)
    cs=torch.cumsum(sp, dim=-1); mask=cs>top_p; mask[...,0]=False
    sp=sp.masked_fill(mask,0.0); sp=sp/sp.sum(dim=-1,keepdim=True).clamp(min=1e-8)
    idx=torch.multinomial(sp,1)
    return si.gather(-1,idx).squeeze(-1)

def parse_ts(ts):
    try: n,d=str(ts).split("/"); return int(n),int(d)
    except: return 4,4

def bars_from_seconds(duration_s,bpm,ts):
    n,d=parse_ts(ts); qpb=n*(4.0/d); return (duration_s*bpm)/(60.0*qpb)

def quantize(v,allowed): return min(allowed,key=lambda x:abs(x-v))

def build_ctrl_prefix_ids(stoi, ctrl_vocab, tempo_bpm, duration_seconds, time_signature, key_opt):
    tconf=ctrl_vocab["tempo_bpm"]
    qbpm=max(tconf["min"],min(tconf["max"], int(round(tempo_bpm/tconf["step"])*tconf["step"])))
    t_tok=f"<CTRL_TEMPO_{qbpm}>"
    bars=bars_from_seconds(duration_seconds, qbpm, time_signature)
    qbars=quantize(bars, ctrl_vocab["length_bars"]["allowed"])
    len_tok=f"<CTRL_LEN_{qbars}BARS>"
    ts_tok=f"<CTRL_TS_{time_signature.replace('/','_')}>"
    ts_ok=set(ctrl_vocab.get("time_signature",{}).get("tokens",[]))
    ctrl=[t_tok, len_tok] + ([ts_tok] if ts_tok in ts_ok else [])
    if key_opt:
        kt=f"<CTRL_KEY_{key_opt.replace(' ','')}>"
        if kt in set(ctrl_vocab.get("key",{}).get("tokens",[])): ctrl.append(kt)
    ids=[stoi.get(t, UNK_ID) for t in ctrl]
    return ids, {"bpm":qbpm, "bars_target": qbars}

# ---- limpeza/decodificação (da 10) ----
def ids_to_tokens_str(id_list, itos):
    toks = []
    for tid in id_list:
        s = itos.get(int(tid), "")
        if not s: continue
        if isinstance(s, str) and s.startswith("<CTRL_"):  # CTRL não vai para MIDI direto
            continue
        if s.startswith("Tempo_"):  # aplicaremos BPM externamente quando necessário
            continue
        toks.append(s)
    return toks

def debug_token_counts(str_tokens):
    # contagem simples para sanidade (opcional)
    from collections import Counter
    c = Counter([t.split("_",1)[0] for t in str_tokens])
    kept = {k:v for k,v in c.items() if v>0}
    print("Token groups:", kept)

# --- Miditok v3 (decode e encode) ---
from miditok import TokenizerConfig, REMI, TokSequence
try:
    from miditok.utils import save_midi as mt_save_midi
except Exception:
    mt_save_midi = None

def _build_tokenizer():
    # Config coerente com a etapa 10 (vista no código)
    return REMI(TokenizerConfig(
        beat_res={(0, 4): 8},
        num_velocities=32,
        use_chords=False,
        use_rests=True,
        use_tempos=True,
        use_time_signatures=True,
        use_programs=True,
        one_token_stream_for_programs=True,
    ))

def tokens_ids_to_pretty_midi(id_list, itos, tmp_dir: Path) -> pm.PrettyMIDI:
    # 1) ids -> strings + limpeza
    str_tokens = ids_to_tokens_str(id_list, itos)
    debug_token_counts(str_tokens)

    # 2) decode miditok
    tok = _build_tokenizer()
    seq = TokSequence(tokens=str_tokens)
    midi_obj = tok.decode(seq)
    if isinstance(midi_obj, list):
        midi_obj = midi_obj[0]

    # 3) salvar em arquivo temporário e carregar com pretty_midi
    tmp_mid = tmp_dir / "gen.mid"
    if mt_save_midi is not None:
        mt_save_midi(midi_obj, tmp_mid)
    else:
        ok=False
        for m in ("dump_midi","dump","save_midi","save","write","write_midi"):
            if hasattr(midi_obj, m):
                try:
                    getattr(midi_obj, m)(str(tmp_mid)); ok=True; break
                except Exception:
                    pass
        if not ok and hasattr(midi_obj, "to_bytes"):
            tmp_mid.write_bytes(midi_obj.to_bytes()); ok=True
        if not ok:
            raise RuntimeError("Não sei salvar este objeto MIDI (symusic).")
    return pm.PrettyMIDI(str(tmp_mid))

def midi_to_ids_for_primer(primer_pm: pm.PrettyMIDI, stoi) -> List[int]:
    tok = _build_tokenizer()
    # miditok espera um objeto MIDI (symusic) — gravamos temporariamente e reabrimos via miditok
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)/"primer.mid"
        primer_pm.write(str(tmp))
        # miditok consegue carregar caminho diretamente
        seqs = tok.encode(str(tmp))
    if isinstance(seqs, list) and len(seqs)>0:
        seq = seqs[0]
        tokens = list(seq.tokens)
    else:
        tokens = list(seqs.tokens) if hasattr(seqs, "tokens") else []

    # filtra tokens de tempo/ctrl para o primer (evita poluir o prefixo)
    tokens = [t for t in tokens if isinstance(t, str) and not t.startswith("<CTRL_") and not t.startswith("Tempo_")]

    ids = [stoi.get(t if isinstance(t, str) else str(t), UNK_ID) for t in tokens]
    return ids

def enforce_bpm_and_crop_file(midi_path: Path, bpm: int, duration_seconds: float):
    m=mido.MidiFile(midi_path)
    for tr in m.tracks:
        tr[:]=[msg for msg in tr if not (msg.is_meta and msg.type=="set_tempo")]
    new=mido.MidiFile(ticks_per_beat=m.ticks_per_beat)
    t=mido.MidiTrack(); t.append(mido.MetaMessage("set_tempo", tempo=int(60_000_000/max(1,bpm)), time=0))
    new.tracks.append(t)
    for tr in m.tracks: new.tracks.append(tr.copy())
    uspb=60_000_000/bpm; us_per_tick=uspb/new.ticks_per_beat; limit=int(duration_seconds*1_000_000)
    for i,tr in enumerate(new.tracks):
        acc=0; out=[]
        for msg in tr:
            acc += int(msg.time*us_per_tick)
            if acc<=limit: out.append(msg)
            else: break
        if not out or out[-1].type!="end_of_track":
            out.append(mido.MetaMessage("end_of_track", time=0))
        new.tracks[i]=mido.MidiTrack(out)
    new.save(midi_path)

def generate_ids_autoregressive(model, itos, init_ids, prefix_emb,
             max_new_tokens, top_p, temperature, stop_on_bars, device,
             min_tokens_before_eos=64,
             forbid_ctrl_ids=None,
             banned_prog_ids=None,
             ban_drum=False,
             drum_pitch_ids=None):
    model.eval()
    out = torch.tensor(init_ids, dtype=torch.long, device=device).unsqueeze(0)
    bars = 0
    steps = 0

    # Em CUDA, o forward pode sair em float16 (autocast). Vamos sempre
    # fazer MASK + sampling em float32 para evitar overflow.
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=(device == "cuda")):
        for _ in range(max_new_tokens):
            idx = out[:, -model.cfg.seq_len + prefix_emb.size(1):]
            logits = model.forward_with_prefix(idx, prefix_emb)  # possivelmente float16

            # Pegamos apenas o último passo e forçamos float32 antes de mascarar
            next_logits = logits[:, -1, :].clone().float()

            # Valor de máscara seguro p/ float32 (em vez de -1e9 em float16)
            mask_val = -1e4

            # Máscaras opcionais
            if forbid_ctrl_ids:
                next_logits[:, forbid_ctrl_ids] = mask_val
            if banned_prog_ids:
                next_logits[:, banned_prog_ids] = mask_val
            if ban_drum and drum_pitch_ids:
                next_logits[:, drum_pitch_ids] = mask_val

            # Evita EOS muito cedo
            if steps < min_tokens_before_eos:
                next_logits[:, EOS_ID] = mask_val

            # Sampling (top-p + temperatura) já espera float32
            next_id = softmax_sample(next_logits.squeeze(0), temperature, top_p).unsqueeze(0)

            s = itos.get(int(next_id), "")
            if isinstance(s, str) and s.startswith("Bar"):
                bars += 1
                if stop_on_bars and bars >= stop_on_bars:
                    out = torch.cat([out, next_id.view(1, 1)], dim=1)
                    break

            out = torch.cat([out, next_id.view(1, 1)], dim=1)
            steps += 1
            if int(next_id) == EOS_ID:
                break

    return out[0].tolist()

# =========================
# Geração segmental (independente da 10)
# =========================

def generate_segment_with_softprompt(
    *,
    primer_midi: pm.PrettyMIDI,          # primer (tail de A por N barras)
    target_seconds: float,               # duração aproximada para o trecho novo (substituto de B)
    tempo_bpm: int,
    time_signature: str,
    prompt_text: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    allow_programs: Optional[str],
    ban_drums: bool,
    ctrl_vocab_path: Path,
    bin_dir: Path,
    soft_ckpt_path: Path,
) -> pm.PrettyMIDI:

    device = "cuda" if torch.cuda.is_available() else "cpu"
    stoi, itos = load_vocab(bin_dir)

    # coletores auxiliares (igual à 10)
    def collect_ids(prefix):
        return [tid for s, tid in stoi.items() if isinstance(s, str) and s.startswith(prefix)]

    forbid_ctrl_ids = [tid for tid, s in itos.items() if isinstance(s, str) and s.startswith("<CTRL_")]
    drum_pitch_ids  = collect_ids("PitchDrum_")

    banned_prog_ids = []
    if allow_programs:
        allowed_set = {int(x) for x in allow_programs.split(",") if x.strip() != ""}
        for s, tid in stoi.items():
            if isinstance(s, str) and s.startswith("Program_"):
                try:
                    val = int(s.split("_", 1)[1])   # aceita -1 (drums)
                    if val not in allowed_set:
                        banned_prog_ids.append(tid)
                except:
                    pass

    ctrl_vocab = json.loads(Path(ctrl_vocab_path).read_text(encoding="utf-8"))

    # carrega checkpoint de soft-prompt
    ck = torch.load(soft_ckpt_path, map_location="cpu")
    cfg = GPTConfig(**ck["cfg"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ck["gpt"], strict=True)
    n_soft = ck["n_soft"]; d_model = ck["cfg"]["d_model"]; dim_text = ck["dim_text"]

    mapper = SoftPromptMapper(dim_text, d_model, n_soft).to(device)
    mapper.load_state_dict(ck["mapper"], strict=True)
    mapper.eval(); model.eval()

    # embedding textual -> soft prompt
    enc = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    emb = enc.encode([prompt_text], normalize_embeddings=True)
    emb = torch.from_numpy(np.asarray(emb, dtype=np.float32)).to(device)
    prefix_emb = mapper(emb)  # [1, n_soft, d_model]

    # prefixo de controle (ids)
    ctrl_ids, info = build_ctrl_prefix_ids(stoi, ctrl_vocab, tempo_bpm, target_seconds, time_signature, None)

    # primer ids (a partir do primer_midi real)
    primer_ids = midi_to_ids_for_primer(primer_midi, stoi)

    # inicia sequência
    init_ids = [BOS_ID] + ctrl_ids + primer_ids

    # amostragem
    full_ids = generate_ids_autoregressive(
        model, itos, init_ids, prefix_emb,
        max_new_tokens, top_p, temperature, info["bars_target"], device,
        min_tokens_before_eos=64,
        forbid_ctrl_ids=forbid_ctrl_ids,
        banned_prog_ids=banned_prog_ids,
        ban_drum=ban_drums,
        drum_pitch_ids=drum_pitch_ids
    )

    musical = full_ids[len(init_ids):]
    if musical and musical[-1] == EOS_ID:
        musical = musical[:-1]

    # decodifica ids -> PrettyMIDI
    with tempfile.TemporaryDirectory() as td:
        pm_gen = tokens_ids_to_pretty_midi(musical, itos, Path(td))
        # crop + enforce bpm no arquivo temporário (igual comportamento da 10)
        tmp_path = Path(td)/"crop.mid"
        pm_gen.write(str(tmp_path))
        enforce_bpm_and_crop_file(tmp_path, info["bpm"], target_seconds)
        pm_final = pm.PrettyMIDI(str(tmp_path))
    return pm_final

# =========================
# Costura
# =========================

def _mix_sequence_with_crossfade(gen_mid: pm.PrettyMIDI, after_mid: pm.PrettyMIDI,
                                 crossfade_seconds: float) -> pm.PrettyMIDI:
    """ Crossfade em velocity na janela [0, crossfade_seconds]. """
    out = pm.PrettyMIDI(resolution=gen_mid.resolution)

    def deep_copy(src: pm.PrettyMIDI) -> pm.PrettyMIDI:
        dst = pm.PrettyMIDI(resolution=src.resolution)
        for inst in src.instruments:
            ni = pm.Instrument(program=inst.program, is_drum=inst.is_drum, name=inst.name)
            for n in inst.notes:
                ni.notes.append(pm.Note(n.velocity, n.pitch, n.start, n.end))
            for b in inst.pitch_bends:
                ni.pitch_bends.append(pm.PitchBend(b.pitch, b.time))
            for c in inst.control_changes:
                ni.control_changes.append(pm.ControlChange(c.number, c.value, c.time))
            dst.instruments.append(ni)
        return dst

    g = deep_copy(gen_mid)
    a = deep_copy(after_mid)

    if crossfade_seconds > 1e-6:
        _scale_velocities(g, max(0.0, g.get_end_time() - crossfade_seconds), g.get_end_time(), 1.0, 0.0)
        _scale_velocities(a, 0.0, min(crossfade_seconds, a.get_end_time()), 0.0, 1.0)

    # concat g + a (com shift de a)
    offset = g.get_end_time()
    for inst in a.instruments:
        si = pm.Instrument(program=inst.program, is_drum=inst.is_drum, name=inst.name)
        for n in inst.notes:
            si.notes.append(pm.Note(n.velocity, n.pitch, n.start + offset, n.end + offset))
        for b in inst.pitch_bends:
            si.pitch_bends.append(pm.PitchBend(b.pitch, b.time + offset))
        for c in inst.control_changes:
            si.control_changes.append(pm.ControlChange(c.number, c.value, c.time + offset))
        out.instruments.append(si)
    for inst in g.instruments:
        out.instruments.append(inst)
    return out

# =========================
# MAIN
# =========================

def main():
    ap = argparse.ArgumentParser()

    # Janela de edição
    ap.add_argument("--start_seconds", type=float, required=True)
    ap.add_argument("--end_seconds",   type=float, required=True)
    ap.add_argument("--source_midi",   type=str,   required=True)
    ap.add_argument("--context_bars_before", type=int, default=4)
    ap.add_argument("--context_bars_after",  type=int, default=2)

    # Saída
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--out_name", type=str, default="edited.mid")

    # ===== Parâmetros criativos (iguais aos de 10, mas usados aqui localmente) =====
    ap.add_argument("--bin_dir",    type=str, required=True)
    ap.add_argument("--soft_ckpt",  type=str, required=True)
    ap.add_argument("--ctrl_vocab", type=str, required=True)
    ap.add_argument("--prompt_text", type=str, default="")
    ap.add_argument("--tempo_bpm", type=int, default=120)
    ap.add_argument("--time_signature", type=str, default="4/4")
    ap.add_argument("--temperature", type=float, default=1.2)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_new_tokens", type=int, default=3000)
    ap.add_argument("--allow_programs", type=str, default="")
    ap.add_argument("--ban_drums", action="store_true")

    args = ap.parse_args()

    # Carregar MIDI fonte e validar janela
    src_path = Path(args.source_midi)
    if not src_path.exists():
        raise FileNotFoundError(src_path)
    pm_src = pm.PrettyMIDI(str(src_path))
    dur_total = pm_src.get_end_time()

    start_s = max(0.0, float(args.start_seconds))
    end_s   = min(float(args.end_seconds), dur_total)
    if end_s <= start_s:
        raise ValueError("end_seconds deve ser > start_seconds e dentro da duração do MIDI.")

    # Barras e regiões
    bars = _bar_boundaries(pm_src)

    # Índice do compasso onde começa o trecho a editar
    bar_idx = max(0, int(np.searchsorted(bars, start_s, side="right") - 1))
    primer_start_bar = max(0, bar_idx - max(0, int(args.context_bars_before)))
    primer_t0 = bars[primer_start_bar]
    primer_t1 = start_s

    after_start_bar = max(0, int(np.searchsorted(bars, end_s, side="right") - 1))
    after_end_bar   = min(len(bars) - 1, after_start_bar + max(0, int(args.context_bars_after)))
    after_t0 = end_s
    after_t1 = bars[after_end_bar] if after_end_bar < len(bars) else dur_total
    if after_t1 <= after_t0:
        after_t1 = min(end_s + 2.0, dur_total)  # fallback de 2s

    # Recortes
    midi_A          = _clip_midi(pm_src, 0.0, start_s)
    midi_primer     = _clip_midi(pm_src, primer_t0, primer_t1)
    midi_B_orig     = _clip_midi(pm_src, start_s, end_s)
    midi_after_head = _clip_midi(pm_src, after_t0, after_t1)
    midi_C_tail     = _clip_midi(pm_src, after_t1, dur_total)

    target_seconds     = midi_B_orig.get_end_time()          # duração alvo do trecho gerado
    crossfade_seconds  = midi_after_head.get_end_time()      # janela de costura

    # ===== Geração independente (copiada/adaptada da 10) =====
    pm_gen_full = generate_segment_with_softprompt(
        primer_midi=midi_primer,
        target_seconds=float(target_seconds),
        tempo_bpm=int(args.tempo_bpm),
        time_signature=str(args.time_signature),
        prompt_text=str(args.prompt_text),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        max_new_tokens=int(args.max_new_tokens),
        allow_programs=(args.allow_programs if args.allow_programs.strip() else None),
        ban_drums=bool(args.ban_drums),
        ctrl_vocab_path=Path(args.ctrl_vocab),
        bin_dir=Path(args.bin_dir),
        soft_ckpt_path=Path(args.soft_ckpt),
    )

    # Ajuste fino de duração (corta se exceder — redundante pois já cropamos no arquivo)
    # gen_len = pm_gen_full.get_end_time()
    # if gen_len > target_seconds + 1e-3:
    #     pm_gen_full = _clip_midi(pm_gen_full, 0.0, target_seconds)


    # === FADES (apenas nas bordas) ===
    # gen_len = pm_gen_full.get_end_time()
    # if crossfade_seconds > 1e-6:
    #     # fade-out no fim do GERADO [gen_len - cf, gen_len]
    #     _scale_velocities(pm_gen_full,
    #                     max(0.0, gen_len - crossfade_seconds),
    #                     gen_len,
    #                     1.0, 0.0)
    #     # fade-in no início do AFTER [0, cf]
    #     _scale_velocities(midi_after_head,
    #                     0.0,
    #                     min(crossfade_seconds, midi_after_head.get_end_time()),
    #                     0.0, 1.0)



    # # === MONTAGEM FINAL (alinhada ao original) ===
    # final = pm.PrettyMIDI(resolution=pm_src.resolution)

    # def append_at(dst: pm.PrettyMIDI, src: pm.PrettyMIDI, t_offset: float):
    #     for inst in src.instruments:
    #         ni = pm.Instrument(program=inst.program, is_drum=inst.is_drum, name=inst.name)
    #         for n in inst.notes:
    #             ni.notes.append(pm.Note(n.velocity, n.pitch, n.start + t_offset, n.end + t_offset))
    #         for b in inst.pitch_bends:
    #             ni.pitch_bends.append(pm.PitchBend(b.pitch, b.time + t_offset))
    #         for c in inst.control_changes:
    #             ni.control_changes.append(pm.ControlChange(c.number, c.value, c.time + t_offset))
    #         dst.instruments.append(ni)

    # # A em 0..start_s
    # append_at(final, midi_A, 0.0)

    # # GERADO exatamente em start..end (mesma duração do trecho editado)
    # append_at(final, pm_gen_full, start_s)

    # # Cabeça de costura do original exatamente em end_s..after_t1
    # append_at(final, midi_after_head, end_s)

    # # Cauda do C exatamente em after_t1..fim
    # append_at(final, midi_C_tail, after_t1)

    # out_dir  = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    # out_midi = out_dir / args.out_name
    # _write_midi(final, out_midi)
    # print("OK:", out_midi.as_posix())



    # === FADES + SOBREPOSIÇÃO ENTRE GERADO E AFTER ===
    gen_len = pm_gen_full.get_end_time()
    after_len = midi_after_head.get_end_time()

    # escolha do overlap real (ex.: até 250 ms), limitado para não ultrapassar os trechos
    xfade = min(0.25, crossfade_seconds, max(0.0, gen_len - 1e-3), max(0.0, after_len - 1e-3))

    if xfade > 1e-6:
        # fade-out no fim do GERADO [gen_len - xfade, gen_len]
        _scale_velocities(
            pm_gen_full,
            max(0.0, gen_len - xfade),
            gen_len,
            1.0, 0.0
        )
        # fade-in no início do AFTER [0, xfade]
        _scale_velocities(
            midi_after_head,
            0.0,
            min(xfade, after_len),
            0.0, 1.0
        )

    # === MONTAGEM FINAL (alinhada ao original) ===
    final = pm.PrettyMIDI(resolution=pm_src.resolution)

    def append_at(dst: pm.PrettyMIDI, src: pm.PrettyMIDI, t_offset: float):
        for inst in src.instruments:
            ni = pm.Instrument(program=inst.program, is_drum=inst.is_drum, name=inst.name)
            for n in inst.notes:
                ni.notes.append(pm.Note(n.velocity, n.pitch, n.start + t_offset, n.end + t_offset))
            for b in inst.pitch_bends:
                ni.pitch_bends.append(pm.PitchBend(b.pitch, b.time + t_offset))
            for c in inst.control_changes:
                ni.control_changes.append(pm.ControlChange(c.number, c.value, c.time + t_offset))
            dst.instruments.append(ni)

    # 1) A em 0..start_s (inalterado)
    append_at(final, midi_A, 0.0)

    # 2) GERADO exatamente em start..end
    append_at(final, pm_gen_full, start_s)

    # 3) AFTER com SOBREPOSIÇÃO: começa em (end_s - xfade)
    #    (se xfade==0, vira corte seco em end_s)
    append_at(final, midi_after_head, end_s - xfade)

    # 4) Cauda C em after_t1..fim (mantém posição original)
    append_at(final, midi_C_tail, after_t1)

    out_dir  = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_midi = out_dir / args.out_name
    _write_midi(final, out_midi)
    print("OK:", out_midi.as_posix())


    # Costura: gen + after_head (com fade) + resto de C
    # stitched = _mix_sequence_with_crossfade(pm_gen_full, midi_after_head, crossfade_seconds)
    

    # Final: A + stitched + C_tail
    # final = pm.PrettyMIDI(resolution=pm_src.resolution)

    # def append_at(dst: pm.PrettyMIDI, src: pm.PrettyMIDI, t_offset: float):
    #     for inst in src.instruments:
    #         ni = pm.Instrument(program=inst.program, is_drum=inst.is_drum, name=inst.name)
    #         for n in inst.notes:
    #             ni.notes.append(pm.Note(n.velocity, n.pitch, n.start + t_offset, n.end + t_offset))
    #         for b in inst.pitch_bends:
    #             ni.pitch_bends.append(pm.PitchBend(b.pitch, b.time + t_offset))
    #         for c in inst.control_changes:
    #             ni.control_changes.append(pm.ControlChange(c.number, c.value, c.time + t_offset))
    #         dst.instruments.append(ni)

    # offset = 0.0
    # append_at(final, midi_A, offset)
    # offset += midi_A.get_end_time()

    # append_at(final, stitched, offset)
    # offset += stitched.get_end_time()

    # append_at(final, midi_C_tail, offset)

    # out_dir  = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    # out_midi = out_dir / args.out_name
    # _write_midi(final, out_midi)

    # print("OK:", out_midi.as_posix())

if __name__ == "__main__":
    main()
