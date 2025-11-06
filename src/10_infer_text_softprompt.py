# 10_infer_text_softprompt.py — inferência com soft-prompt textual (autocontido)
import argparse, json, math, random, subprocess
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import mido

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0,1,2,3

def load_tokens_index(tokens_jsonl: Path):
    idx = {}
    with open(tokens_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            rec = json.loads(line)
            mp = rec.get("midi_path")
            toks = rec.get("tokens", [])
            if toks and isinstance(toks[0], (list, tuple)):
                toks = toks[0]
            if mp: idx[mp] = toks
    return idx

def strip_ctrl_prefix(token_str_list):
    i0 = 0
    for t in token_str_list:
        if isinstance(t, str) and t.startswith("<CTRL_"):
            i0 += 1
        else:
            break
    return token_str_list[i0:]

def retrieve_best_midi(prompt, captions_csv: Path, tokens_idx: dict):
    df = pd.read_csv(captions_csv)
    df = df[df["midi_path"].isin(tokens_idx.keys())].copy()
    if len(df) == 0: return None
    caps = df["caption"].fillna("").astype(str).tolist()
    vec  = TfidfVectorizer(max_features=20000).fit(caps + [prompt])
    M    = vec.transform(caps)
    q    = vec.transform([prompt])
    j    = int(cosine_similarity(M, q).ravel().argmax())
    return df.iloc[j]

def make_primer_ids(row, tokens_idx, stoi, bars=8, tokens_override=0):
    toks = strip_ctrl_prefix(tokens_idx.get(row["midi_path"], []))
    if tokens_override and tokens_override > 0:
        toks = toks[:tokens_override]
    else:
        out, b = [], 0
        for t in toks:
            out.append(t)
            if isinstance(t, str) and t.startswith("Bar"):
                b += 1
                if b >= bars: break
        toks = out
    return [stoi.get(t if isinstance(t, str) else str(t), 3) for t in toks]  # 3 = UNK_ID


def ids_to_tokens_str(id_list, itos):
    """Converte ids -> strings do seu vocab.json, limpa controles e normaliza alguns tokens."""
    toks = []
    for tid in id_list:
        s = itos.get(int(tid), "")
        if not s:
            continue
        if isinstance(s, str) and s.startswith("<CTRL_"):
            continue  # já controlamos BPM/len fora
        # if s == "Bar_None":
        #     s = "Bar"  # normaliza
        if s.startswith("Tempo_"):
            continue  # vamos forçar BPM depois
        toks.append(s)

    bad = [t for t in toks if "_" not in t]
    if bad:
        print("TOKENS sem '_':", bad[:20])
        # opcional: raise ValueError("Tokens sem '_' encontrados")

    return toks

def debug_token_counts(str_tokens):
    from collections import Counter
    c = Counter(t.split("_", 1)[0] for t in str_tokens)
    print("DBG token tipos:", dict(c))


# ---------------- GPT com suporte a prefixo (igual ao treino) ----------------
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
    def __init__(self,cfg): super().__init__(); self.fc1=nn.Linear(cfg.d_model,cfg.d_ff); self.fc2=nn.Linear(cfg.d_ff,cfg.d_model); self.drop=nn.Dropout(cfg.dropout); self.act=nn.GELU()
    def forward(self,x): return self.drop(self.fc2(self.act(self.fc1(x))))

class Block(nn.Module):
    def __init__(self,cfg): super().__init__(); self.ln1=nn.LayerNorm(cfg.d_model); self.attn=CausalSelfAttention(cfg); self.ln2=nn.LayerNorm(cfg.d_model); self.mlp=MLP(cfg)
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

# ---------------- SoftPromptMapper (igual ao treino) ----------------
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

# ---------------- util ----------------
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
    sp[mask]=0; sp=sp/sp.sum(-1,keepdim=True)
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

# --- Miditok v3 (salvar MIDI) ---
from miditok import TokenizerConfig, REMI, TokSequence
try:
    from miditok.utils import save_midi as mt_save_midi
except Exception:
    mt_save_midi = None

def tokens_to_midi_save_from_ids(id_list, itos, out_path: Path):
    # 1) ids -> strings + limpeza
    str_tokens = ids_to_tokens_str(id_list, itos)
    debug_token_counts(str_tokens)  # deve mostrar Pitch/Velocity/Duration > 0

    # 2) constrói tokenizer compatível (mesma config usada na tokenização)
    tok = REMI(TokenizerConfig(
        beat_res={(0, 4): 8},
        num_velocities=32,
        use_chords=False,
        use_rests=True,
        use_tempos=True,
        use_time_signatures=True,
        use_programs=True,
        one_token_stream_for_programs=True,
    ))

    # 3) converte para MIDI
    seq = TokSequence(tokens=str_tokens)   # um único track/stream
    midi_obj = tok.decode(seq) 
    if isinstance(midi_obj, list):
        midi_obj = midi_obj[0]

    # 4) salva
    if mt_save_midi is not None:
        mt_save_midi(midi_obj, out_path)
    else:
        # fallbacks genéricos
        for m in ("dump_midi","dump","save_midi","save","write","write_midi"):
            if hasattr(midi_obj, m):
                try:
                    getattr(midi_obj, m)(str(out_path)); return
                except Exception:
                    pass
        if hasattr(midi_obj, "to_bytes"):
            out_path.write_bytes(midi_obj.to_bytes()); return
        raise RuntimeError("Não sei salvar este objeto MIDI (symusic).")
    
def enforce_bpm_and_crop(midi_path: Path, bpm: int, duration_seconds: float):
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

# ---------------- geração ----------------
def generate(model, itos, init_ids, prefix_emb,
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
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=(device=="cuda")):
        for _ in range(max_new_tokens):
            idx = out[:, -model.cfg.seq_len + prefix_emb.size(1):]
            logits = model.forward_with_prefix(idx, prefix_emb)
            next_logits = logits[:, -1, :].float()  # trabalha em fp32 pra evitar overflow
            mask_val = torch.finfo(next_logits.dtype).min

            if forbid_ctrl_ids:    next_logits[:, forbid_ctrl_ids] = mask_val
            if banned_prog_ids:    next_logits[:, banned_prog_ids] = mask_val
            if ban_drum and drum_pitch_ids:
                next_logits[:, drum_pitch_ids] = mask_val

            # bloqueia EOS nos primeiros N passos
            EOS_ID = 2
            if steps < min_tokens_before_eos:
                next_logits[:, EOS_ID] = mask_val

            # amostra
            next_id = softmax_sample(next_logits.squeeze(0), temperature, top_p).unsqueeze(0)
            s = itos[int(next_id)]

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

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin_dir", required=True)
    ap.add_argument("--soft_ckpt", required=True, help="checkpoint do soft-prompt (softprompt_epXX.pt)")
    ap.add_argument("--ctrl_vocab", required=True)
    ap.add_argument("--out_dir", default="data/infer_out")
    ap.add_argument("--prompt_text", required=True)
    ap.add_argument("--tempo_bpm", type=int, default=120)
    ap.add_argument("--duration_seconds", type=float, default=30.0)
    ap.add_argument("--time_signature", type=str, default="4/4")
    ap.add_argument("--key", type=str, default="")
    ap.add_argument("--max_new_tokens", type=int, default=1500)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--allow_programs", type=str, default="", help="ex: 40,41,42  (GM programs permitidos)")
    ap.add_argument("--ban_drums", action="store_true", help="bloqueia PitchDrum_*")
    ap.add_argument("--force_program", type=int, default=-1, help="se >=0, injeta Program_N logo após os CTRLs")
    ap.add_argument("--captions_csv", type=str, default="", help="CSV com midi_path e caption (ex.: data/splits/train_5k.csv)")
    ap.add_argument("--tokens_jsonl", type=str, default="", help="JSONL com tokens do split (ex.: data/tokens/train_5k.jsonl)")
    ap.add_argument("--primer_bars", type=int, default=8, help="nº de barras do primer (0 desliga)")
    ap.add_argument("--primer_tokens", type=int, default=0, help="override por nº de tokens (se >0, ignora barras)")
    ap.add_argument("--out_name", type=str, default="gen_text.mid",
                    help="nome do arquivo MIDI de saída (ex.: gen_01.mid)")

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    bin_dir = Path(args.bin_dir); out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    stoi, itos = load_vocab(bin_dir)

    def collect_ids(stoi, prefix):
        return [tid for s, tid in stoi.items() if isinstance(s, str) and s.startswith(prefix)]

    # 1) CTRLs que não podem aparecer após o prefixo
    forbid_ctrl_ids = [tid for tid, s in enumerate([None]*(max(stoi.values())+1))]
    forbid_ctrl_ids = [tid for tid, s in itos.items() if isinstance(s, str) and s.startswith("<CTRL_")]

    # 2) PitchDrum_* (para banir bateria)
    drum_pitch_ids = collect_ids(stoi, "PitchDrum_")

    # 3) Program_* NÃO permitidos (se --allow_programs foi passado)
    banned_prog_ids = []
    if args.allow_programs:
        allowed_set = {int(x) for x in args.allow_programs.split(",") if x.strip() != ""}
        for s, tid in stoi.items():
            if isinstance(s, str) and s.startswith("Program_"):
                try:
                    val = int(s.split("_", 1)[1])   # aceita -1 (drums)
                    if val not in allowed_set:
                        banned_prog_ids.append(tid)
                except:
                    pass

    ctrl_vocab = json.loads(Path(args.ctrl_vocab).read_text(encoding="utf-8"))

    # carrega checkpoint de soft-prompt
    ck = torch.load(args.soft_ckpt, map_location="cpu")
    cfg = GPTConfig(**ck["cfg"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ck["gpt"], strict=True)
    n_soft = ck["n_soft"]; d_model = ck["cfg"]["d_model"]; dim_text = ck["dim_text"]

    mapper = SoftPromptMapper(dim_text, d_model, n_soft).to(device)
    mapper.load_state_dict(ck["mapper"], strict=True)
    mapper.eval(); model.eval()

    # embedding do texto
    enc = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    emb = enc.encode([args.prompt_text], normalize_embeddings=True)
    emb = torch.from_numpy(np.asarray(emb, dtype=np.float32)).to(device)
    prefix_emb = mapper(emb)  # [1, n_soft, d_model]

    # # prefixo de controle (ids)
    # ctrl_ids, info = build_ctrl_prefix_ids(stoi, ctrl_vocab, args.tempo_bpm, args.duration_seconds, args.time_signature, args.key or None)
    # primer_ids = []
    # if args.captions_csv and args.tokens_jsonl and args.prompt_text and (args.primer_bars > 0 or args.primer_tokens > 0):
    #     idx = load_tokens_index(Path(args.tokens_jsonl))
    #     best = retrieve_best_midi(args.prompt_text, Path(args.captions_csv), idx)
    #     if best is not None:
    #         primer_ids = make_primer_ids(best, idx, stoi, bars=args.primer_bars, tokens_override=args.primer_tokens)
    #         print("Primer de:", best["midi_path"])
    #     else:
    #         print("Sem primer (retrieval não achou candidato).")
    # init_ids = [BOS_ID] + ctrl_ids
    # if args.force_program >= 0:
    #     tok_force = f"Program_{args.force_program}"
    #     init_ids.append(stoi.get(tok_force, UNK_ID))
    # # depois acrescente o primer, se houver
    # init_ids += primer_ids

    # prefixo de controle (ids)
    ctrl_ids, info = build_ctrl_prefix_ids(stoi, ctrl_vocab, args.tempo_bpm, args.duration_seconds, args.time_signature, args.key or None)
    primer_ids = []
    primer_programs = set()

    # -------------------------------------------------------
    # 1. Seleciona primer se solicitado (mantém sua lógica)
    # -------------------------------------------------------
    if args.captions_csv and args.tokens_jsonl and args.prompt_text and (args.primer_bars > 0 or args.primer_tokens > 0):
        idx = load_tokens_index(Path(args.tokens_jsonl))
        best = retrieve_best_midi(args.prompt_text, Path(args.captions_csv), idx)
        if best is not None:
            primer_ids = make_primer_ids(best, idx, stoi, bars=args.primer_bars, tokens_override=args.primer_tokens)
            print("Primer de:", best["midi_path"])
            # coleta instrumentos do primer
            toks = strip_ctrl_prefix(idx.get(best["midi_path"], []))
            for t in toks:
                if isinstance(t, str) and t.startswith("Program_"):
                    try:
                        primer_programs.add(int(t.split("_", 1)[1]))
                    except:
                        pass
        else:
            print("Sem primer (retrieval não achou candidato).")

    # -------------------------------------------------------
    # 2. Define instrumentos permitidos (herança ou fallback)
    # -------------------------------------------------------
    allowed_programs = set()
    if args.allow_programs:
        allowed_programs = {int(x) for x in args.allow_programs.split(",") if x.strip() != ""}
    elif primer_programs:
        allowed_programs = primer_programs
        print(f"Instrumentos herdados do primer: {sorted(list(allowed_programs))}")
    else:
        # Sem primer e sem instrumentos -> tenta achar um primer só pra pegar os instrumentos
        if args.captions_csv and args.tokens_jsonl:
            idx = load_tokens_index(Path(args.tokens_jsonl))
            best_fallback = retrieve_best_midi(args.prompt_text, Path(args.captions_csv), idx)
            if best_fallback is not None:
                toks = strip_ctrl_prefix(idx.get(best_fallback["midi_path"], []))
                for t in toks:
                    if isinstance(t, str) and t.startswith("Program_"):
                        try:
                            allowed_programs.add(int(t.split("_", 1)[1]))
                        except:
                            pass
                if allowed_programs:
                    print(f"Instrumentos herdados de primer alternativo ({best_fallback['midi_path']}): {sorted(list(allowed_programs))}")

    # -------------------------------------------------------
    # 3. Monta lista de programas banidos com base nos permitidos
    # -------------------------------------------------------
    banned_prog_ids = []
    if allowed_programs:
        for s, tid in stoi.items():
            if isinstance(s, str) and s.startswith("Program_"):
                try:
                    val = int(s.split("_", 1)[1])
                    if val not in allowed_programs:
                        banned_prog_ids.append(tid)
                except:
                    pass

    # -------------------------------------------------------
    # 4. Monta sequência inicial (prefixo)
    # -------------------------------------------------------
    init_ids = [BOS_ID] + ctrl_ids
    if args.force_program >= 0:
        tok_force = f"Program_{args.force_program}"
        init_ids.append(stoi.get(tok_force, UNK_ID))
    # adiciona primer (se houver)
    init_ids += primer_ids


    full = generate(
                    model, itos, init_ids, prefix_emb,
                    args.max_new_tokens, args.top_p, args.temperature, info["bars_target"], device,
                    min_tokens_before_eos=64,
                    forbid_ctrl_ids=forbid_ctrl_ids,
                    banned_prog_ids=banned_prog_ids,
                    ban_drum=args.ban_drums,
                    drum_pitch_ids=drum_pitch_ids
                  )   
    musical = full[len(init_ids):]
    if musical and musical[-1]==EOS_ID: musical = musical[:-1]

    mid_path = out_dir / args.out_name


    print("Preview tokens:", [itos.get(t, "?") for t in (full[:40])])
    tokens_to_midi_save_from_ids(musical, itos, mid_path)

    def count_notes(p):
        m = mido.MidiFile(p)
        return sum(1 for tr in m.tracks for msg in tr if msg.type == "note_on" and msg.velocity > 0)

    print("Notas no MIDI (antes do crop):", count_notes(mid_path))


    enforce_bpm_and_crop(mid_path, info["bpm"], args.duration_seconds)
    print("MIDI:", mid_path.as_posix())

if __name__ == "__main__":
    main()
