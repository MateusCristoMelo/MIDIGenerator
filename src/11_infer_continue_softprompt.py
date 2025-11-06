# 11_infer_continue_softprompt.py
# Continuação por soft-prompt usando tail do MIDI de origem como primer

import argparse, json, math, tempfile
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pretty_midi as pm
import mido

import torch, torch.nn as nn, torch.nn.functional as F
from sentence_transformers import SentenceTransformer

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
        grid = np.arange(0.0, dur + 1e-6, 2.0).tolist()
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

def _write_midi(pmidi: pm.PrettyMIDI, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pmidi.write(str(path))

# =========================
# Modelo/soft-prompt (copiado/adaptado da 10/12)
# =========================

PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3

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
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m,nn.Embedding):
            nn.init.normal_(m.weight,0.0,0.02)
    def forward_with_prefix(self, idx, prefix_emb=None):
        B,T=idx.shape
        if prefix_emb is not None:
            P=prefix_emb.size(1); total=P+T
            assert total<=self.cfg.seq_len
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
        self.proj=nn.Sequential(
            nn.Linear(dim_text,2*d_model),
            nn.GELU(),
            nn.Linear(2*d_model,n_soft*d_model)
        )
        self.n_soft=n_soft; self.d_model=d_model
    def forward(self, emb):  # [B, dim_text]
        x=self.proj(emb); return x.view(-1,self.n_soft,self.d_model)

def load_vocab(bin_dir: Path):
    stoi=json.loads((bin_dir/"vocab.json").read_text(encoding="utf-8"))
    itos={int(v):k for k,v in stoi.items()}
    return stoi, itos

def softmax_sample(logits, temperature=1.0, top_p=0.9, top_k: int = 0):
    logits=logits.float()
    if temperature>0: logits=logits/temperature
    probs=logits.softmax(-1)

    if top_k and top_k>0:
        # top-k truncation
        sp, si=torch.topk(probs, k=min(top_k, probs.size(-1)), dim=-1)
        sp = sp / sp.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        idx=torch.multinomial(sp,1)
        return si.gather(-1,idx).squeeze(-1)

    # nucleus (top-p)
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

# ---------- miditok encode/decode ----------
from miditok import TokenizerConfig, REMI, TokSequence
try:
    from miditok.utils import save_midi as mt_save_midi
except Exception:
    mt_save_midi = None

def _build_tokenizer():
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

def midi_to_ids(pmidi: pm.PrettyMIDI, stoi) -> List[int]:
    tok = _build_tokenizer()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)/"x.mid"
        pmidi.write(str(tmp))
        seqs = tok.encode(str(tmp))
    if isinstance(seqs, list) and len(seqs)>0:
        seq = seqs[0]
        tokens = list(seq.tokens)
    else:
        tokens = list(seqs.tokens) if hasattr(seqs, "tokens") else []
    tokens = [t for t in tokens if isinstance(t, str) and not t.startswith("<CTRL_")]
    return [stoi.get(t, UNK_ID) for t in tokens]

def ids_to_midi(id_list, itos, bpm: int, target_seconds: float) -> pm.PrettyMIDI:
    # ids -> strings (limpando ctrl)
    str_tokens=[]
    for tid in id_list:
        s=itos.get(int(tid),"")
        if not s: continue
        if s.startswith("<CTRL_"): continue
        if s.startswith("Tempo_"): continue
        str_tokens.append(s)

    tok=_build_tokenizer()
    seq=TokSequence(tokens=str_tokens)
    midi_obj = tok.decode(seq)
    if isinstance(midi_obj, list):
        midi_obj = midi_obj[0]

    # --- salvar robusto ---
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "gen.mid"
        saved = False

        # 1) preferir util do miditok, se disponível
        if mt_save_midi is not None:
            try:
                mt_save_midi(midi_obj, p)
                saved = True
            except Exception:
                saved = False

        # 2) objetos do symusic geralmente expõem to_bytes()
        if not saved and hasattr(midi_obj, "to_bytes"):
            try:
                p.write_bytes(midi_obj.to_bytes())  # type: ignore
                saved = True
            except Exception:
                pass

        # 3) alguns builds expõem dump_midi / write_midi
        if not saved and hasattr(midi_obj, "dump_midi"):
            try:
                midi_obj.dump_midi(str(p))  # type: ignore
                saved = True
            except Exception:
                pass
        if not saved and hasattr(midi_obj, "write_midi"):
            try:
                midi_obj.write_midi(str(p))  # type: ignore
                saved = True
            except Exception:
                pass

        if not saved:
            raise RuntimeError("Falha ao salvar MIDI decodificado (symusic).")

        # força BPM + corta por tempo e volta como PrettyMIDI
        enforce_bpm_and_crop_file(p, bpm, target_seconds)
        return pm.PrettyMIDI(str(p))

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

# =========================
# Geração (autoregressiva)
# =========================

def generate_ids_autoregressive(model, itos, init_ids, prefix_emb,
             max_new_tokens, top_p, temperature, top_k, stop_on_bars, device,
             min_tokens_before_eos=64,
             forbid_ctrl_ids=None,
             banned_prog_ids=None,
             ban_drum=False,
             drum_pitch_ids=None,
             rest_ids=None,
             timeshift_ids=None,
             rest_penalty=0.0,
             ts_penalty=0.0):
    model.eval()
    out = torch.tensor(init_ids, dtype=torch.long, device=device).unsqueeze(0)
    bars = 0
    steps = 0

    with torch.no_grad(), torch.amp.autocast('cuda', enabled=(device == "cuda")):
        for _ in range(max_new_tokens):
            idx = out[:, -model.cfg.seq_len + prefix_emb.size(1):]
            logits = model.forward_with_prefix(idx, prefix_emb)

            # sempre mascarar/amostrar em float32
            next_logits = logits[:, -1, :].clone().float()
            mask_val = -1e4

            if forbid_ctrl_ids:
                next_logits[:, forbid_ctrl_ids] = mask_val
            if banned_prog_ids:
                next_logits[:, banned_prog_ids] = mask_val
            if ban_drum and drum_pitch_ids:
                next_logits[:, drum_pitch_ids] = mask_val
            if steps < min_tokens_before_eos:
                next_logits[:, EOS_ID] = mask_val

            # penalização de “espaços vazios”
            if rest_ids:
                next_logits[:, rest_ids] -= rest_penalty
            if timeshift_ids:
                next_logits[:, timeshift_ids] -= ts_penalty

            next_id = softmax_sample(next_logits.squeeze(0), temperature, top_p, top_k).unsqueeze(0)

            s = itos.get(int(next_id), "")
            if isinstance(s, str) and s.startswith("Bar"):
                bars += 1
                if stop_on_bars and bars >= stop_on_bars:
                    out = torch.cat([out, next_id.view(1,1)], dim=1)
                    break

            out = torch.cat([out, next_id.view(1,1)], dim=1)
            steps += 1
            if int(next_id) == EOS_ID:
                break

    return out[0].tolist()

# =========================
# MAIN
# =========================

def main():
    ap = argparse.ArgumentParser()

    # arquivos/paths
    ap.add_argument("--bin_dir",    type=str, required=True)
    ap.add_argument("--soft_ckpt",  type=str, required=True)
    ap.add_argument("--ctrl_vocab", type=str, required=True)
    ap.add_argument("--out_dir",    type=str, required=True)
    ap.add_argument("--out_name",   type=str, default="continued.mid")

    # prompt/condições
    ap.add_argument("--prompt_text", type=str, default="")
    ap.add_argument("--tempo_bpm",   type=int, default=120)
    ap.add_argument("--time_signature", type=str, default="4/4")
    ap.add_argument("--duration_seconds", type=float, required=True)

    # amostragem
    ap.add_argument("--temperature", type=float, default=1.2)
    ap.add_argument("--top_p",       type=float, default=0.95)
    ap.add_argument("--top_k",       type=int,   default=0)      # 0 = desliga top-k
    ap.add_argument("--max_new_tokens", type=int, default=3000)

    # restrições
    ap.add_argument("--allow_programs", type=str, default="")
    ap.add_argument("--ban_drums", action="store_true")

    # densidade (penalização de vazios)
    ap.add_argument("--rest_penalty",      type=float, default=0.0)
    ap.add_argument("--timeshift_penalty", type=float, default=0.0)

    # primer
    ap.add_argument("--source_midi", type=str, required=True)
    ap.add_argument("--tail_bars",   type=int, required=True)

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # carregar vocabulário/ctrl
    bin_dir = Path(args.bin_dir)
    stoi, itos = load_vocab(bin_dir)
    ctrl_vocab = json.loads(Path(args.ctrl_vocab).read_text(encoding="utf-8"))

    # listas auxiliares para mascaramento/penalização
    def collect_ids(prefix): return [tid for s,tid in stoi.items() if isinstance(s,str) and s.startswith(prefix)]
    forbid_ctrl_ids = [tid for tid, s in itos.items() if isinstance(s, str) and s.startswith("<CTRL_")]
    drum_pitch_ids  = collect_ids("PitchDrum_")
    rest_ids        = collect_ids("Rest_")
    timeshift_ids   = collect_ids("TimeShift_")

    banned_prog_ids = []
    if args.allow_programs:
        allowed_set = {int(x) for x in args.allow_programs.split(",") if x.strip() != ""}
        for s, tid in stoi.items():
            if isinstance(s, str) and s.startswith("Program_"):
                try:
                    val = int(s.split("_", 1)[1])
                    if val not in allowed_set:
                        banned_prog_ids.append(tid)
                except:
                    pass

    # carregar checkpoint/mapper
    ck = torch.load(Path(args.soft_ckpt), map_location="cpu")
    cfg = GPTConfig(**ck["cfg"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ck["gpt"], strict=True)
    n_soft = ck["n_soft"]; d_model = ck["cfg"]["d_model"]; dim_text = ck["dim_text"]

    mapper = SoftPromptMapper(dim_text, d_model, n_soft).to(device)
    mapper.load_state_dict(ck["mapper"], strict=True)
    mapper.eval(); model.eval()

    # embedding textual
    enc = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    emb = enc.encode([args.prompt_text], normalize_embeddings=True)
    emb = torch.from_numpy(np.asarray(emb, dtype=np.float32)).to(device)
    prefix_emb = mapper(emb)  # [1, n_soft, d_model]

    # carregar MIDI de origem e extrair tail_bars como primer
    src_path = Path(args.source_midi)
    if not src_path.exists():
        raise FileNotFoundError(src_path)
    pm_src = pm.PrettyMIDI(str(src_path))
    dur_total = pm_src.get_end_time()
    bars = _bar_boundaries(pm_src)
    # últimos tail_bars compassos
    start_bar = max(0, len(bars)-1 - max(0,int(args.tail_bars)))
    primer_t0 = bars[start_bar]
    primer_t1 = dur_total
    midi_primer = _clip_midi(pm_src, primer_t0, primer_t1)

    # ids de controle (usa duration_seconds)
    ctrl_ids, info = build_ctrl_prefix_ids(stoi, ctrl_vocab, args.tempo_bpm, args.duration_seconds, args.time_signature, None)

    # ids do primer (midi->tokens->ids)
    primer_ids = midi_to_ids(midi_primer, stoi)

    # sequência inicial
    init_ids = [BOS_ID] + ctrl_ids + primer_ids

    # geração
    full_ids = generate_ids_autoregressive(
        model, itos, init_ids, prefix_emb,
        max_new_tokens=args.max_new_tokens,
        top_p=args.top_p,
        temperature=args.temperature,
        top_k=int(args.top_k) if args.top_k>0 else 0,
        stop_on_bars=None,                   # preferimos cortar por tempo ao final
        device=device,
        min_tokens_before_eos=64,
        forbid_ctrl_ids=forbid_ctrl_ids,
        banned_prog_ids=banned_prog_ids,
        ban_drum=bool(args.ban_drums),
        drum_pitch_ids=drum_pitch_ids,
        rest_ids=rest_ids if args.rest_penalty>0 else None,
        timeshift_ids=timeshift_ids if args.timeshift_penalty>0 else None,
        rest_penalty=float(args.rest_penalty),
        ts_penalty=float(args.timeshift_penalty)
    )

    musical = full_ids[len(init_ids):]
    if musical and musical[-1] == EOS_ID:
        musical = musical[:-1]

    # decodifica ids -> PrettyMIDI (força bpm e crop por duração)
    pm_gen = ids_to_midi(musical, itos, bpm=info["bpm"], target_seconds=float(args.duration_seconds))

    # salvar: segmento gerado + concat final
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    seg_name = Path(args.out_name).with_suffix("").name + "_cont_segment.mid"
    out_seg  = out_dir / seg_name
    _write_midi(pm_gen, out_seg)

    # concatena ao original na posição correta (fim de pm_src)
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

    append_at(final, pm_src, 0.0)
    append_at(final, pm_gen, dur_total)

    out_final = out_dir / args.out_name
    _write_midi(final, out_final)

    print("OK: segment", out_seg.as_posix())
    print("OK: final  ", out_final.as_posix())

if __name__ == "__main__":
    main()
