import argparse, json, math, time, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3

# ----------------- Modelo (mesmo GPT do 07, com suporte a prefixo) -----------------
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
    def forward(self, x):
        B,T,C=x.shape
        qkv=self.qkv(x); q,k,v=qkv.split(C,dim=2)
        q=q.view(B,T,self.nh,self.hd).transpose(1,2)
        k=k.view(B,T,self.nh,self.hd).transpose(1,2)
        v=v.view(B,T,self.nh,self.hd).transpose(1,2)
        att=(q@k.transpose(-2,-1))/math.sqrt(self.hd)
        att=att.masked_fill(self.mask[:,:,:T,:T]==0, float('-inf'))
        att=F.softmax(att, dim=-1); att=self.attn_drop(att)
        y=att@v; y=y.transpose(1,2).contiguous().view(B,T,C)
        return self.resid_drop(self.proj(y))

class MLP(nn.Module):
    def __init__(self, cfg): super().__init__(); self.fc1=nn.Linear(cfg.d_model,cfg.d_ff); self.fc2=nn.Linear(cfg.d_ff,cfg.d_model); self.drop=nn.Dropout(cfg.dropout); self.act=nn.GELU()
    def forward(self,x): return self.drop(self.fc2(self.act(self.fc1(x))))

class Block(nn.Module):
    def __init__(self,cfg): super().__init__(); self.ln1=nn.LayerNorm(cfg.d_model); self.attn=CausalSelfAttention(cfg); self.ln2=nn.LayerNorm(cfg.d_model); self.mlp=MLP(cfg)
    def forward(self,x): x = x + self.attn(self.ln1(x)); x = x + self.mlp(self.ln2(x)); return x

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
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:             # <-- ADICIONE ESTA CHECAGEM
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward_with_prefix(self, idx, prefix_emb=None):
        # idx: [B,T]; prefix_emb: [B,P,d_model] ou None
        B,T=idx.shape
        if prefix_emb is not None:
            P = prefix_emb.size(1)
            total = P + T
            assert total <= self.cfg.seq_len, f"comprimento {total} > seq_len {self.cfg.seq_len}"
            pos = torch.arange(total, device=idx.device).unsqueeze(0)
            x_tok = self.tok_emb(idx)
            x = torch.cat([prefix_emb, x_tok], dim=1) + self.pos_emb(pos)
        else:
            pos = torch.arange(T, device=idx.device).unsqueeze(0)
            x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        for blk in self.blocks: x = blk(x)
        x = self.ln_f(x)
        return self.head(x)

# ----------------- Dataset (lê JSONL + embeddings por midi_path) -----------------
class JsonlWithTextDataset(Dataset):
    def __init__(self, jsonl_path: Path, text_parquet: Path, stoi: dict, seq_len: int, n_soft: int):
        self.seq_len = seq_len; self.n_soft = n_soft; self.stoi = stoi
        self.records = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                rec = json.loads(line)
                toks = rec.get("tokens", [])
                if toks and isinstance(toks[0], (list, tuple)): toks = toks[0]
                self.records.append({"midi_path": rec.get("midi_path"), "tokens": toks})
        df = pd.read_parquet(text_parquet)
        self.text_map = {row.midi_path: np.array(row.embed, dtype=np.float32) for _,row in df.iterrows()}
        # fallback: se não achar texto, usa vetor zero
        self.dim = len(next(iter(self.text_map.values()))) if self.text_map else 384

    def __len__(self): return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        toks = r["tokens"]
        # converte tokens -> ids
        ids = [self.stoi.get(t if isinstance(t,str) else str(t), UNK_ID) for t in toks]
        # corta para caber no seq_len depois de adicionar SOFT (n_soft) — lembrando que faremos x=ids[:-1], y=ids[1:]
        max_T = self.seq_len - self.n_soft
        ids = ids[:max_T]
        x = np.array(ids[:-1], dtype=np.int64)
        y = np.array(ids[1:],  dtype=np.int64)

        # embedding do texto (ou zeros)
        emb = self.text_map.get(r["midi_path"], np.zeros((self.dim,), dtype=np.float32))
        return x, y, emb

def collate(batch):
    # pad à esquerda/direita para mesmo T dentro do batch
    xs, ys, es = zip(*batch)
    T = max(len(x) for x in xs)
    xb = np.full((len(xs), T), PAD_ID, dtype=np.int64)
    yb = np.full((len(xs), T), PAD_ID, dtype=np.int64)
    for i,(x,y) in enumerate(zip(xs,ys)):
        xb[i, :len(x)] = x
        yb[i, :len(y)] = y
    eb = np.stack(es).astype(np.float32)  # [B, dim_text]
    return torch.from_numpy(xb), torch.from_numpy(yb), torch.from_numpy(eb)

# ----------------- Soft-prompt mapper (texto_emb -> [n_soft, d_model]) -----------------
class SoftPromptMapper(nn.Module):
    def __init__(self, dim_text: int, d_model: int, n_soft: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim_text, 2*d_model),
            nn.GELU(),
            nn.Linear(2*d_model, n_soft*d_model)
        )
        self.n_soft = n_soft; self.d_model = d_model
    def forward(self, emb):  # emb: [B, dim_text]
        x = self.proj(emb)           # [B, n_soft*d_model]
        return x.view(-1, self.n_soft, self.d_model)  # [B, n_soft, d_model]

# ----------------- Treino -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--val_jsonl", required=True)
    ap.add_argument("--text_train_parquet", required=True)
    ap.add_argument("--text_val_parquet", required=True)
    ap.add_argument("--bin_dir", required=True)   # onde está vocab.json (step 06)
    ap.add_argument("--out_dir", default="runs/text_softprompt")
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--n_soft", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--resume_from", type=str, default=None,
                help="Caminho para um checkpoint .pt anterior para continuar o treino")

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    stoi = json.loads((Path(args.bin_dir)/"vocab.json").read_text(encoding="utf-8"))
    vocab_size = max(int(v) for v in stoi.values()) + 1

    # datasets/loader
    tr_ds = JsonlWithTextDataset(Path(args.train_jsonl), Path(args.text_train_parquet), stoi, args.seq_len, args.n_soft)
    va_ds = JsonlWithTextDataset(Path(args.val_jsonl),   Path(args.text_val_parquet),   stoi, args.seq_len, args.n_soft)
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers, collate_fn=collate, pin_memory=True, drop_last=True)
    va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate, pin_memory=True, drop_last=False)

    # modelo + soft prompt mapper
    cfg = GPTConfig(vocab_size=vocab_size, seq_len=args.seq_len)
    gpt = GPT(cfg).to(device)
    dim_text = tr_ds.dim
    mapper = SoftPromptMapper(dim_text, cfg.d_model, args.n_soft).to(device)

    # ✅ Retomar pesos se --resume_from for fornecido
    if args.resume_from and Path(args.resume_from).exists():
        print(f"🔄 Retomando treino de {args.resume_from}")
        ckpt = torch.load(args.resume_from, map_location=device)
        if "gpt" in ckpt:
            gpt.load_state_dict(ckpt["gpt"], strict=False)
        if "mapper" in ckpt:
            mapper.load_state_dict(ckpt["mapper"], strict=False)
        print("Checkpoint carregado com sucesso.")

    params = list(gpt.parameters()) + list(mapper.parameters())
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler('cuda', enabled=(device=="cuda"))
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    def run_epoch(loader, train=True):
        model = gpt; model.train(train); mapper.train(train)
        total, steps = 0.0, 0
        for xb, yb, eb in loader:
            xb, yb, eb = xb.to(device), yb.to(device), eb.to(device)
            with torch.amp.autocast('cuda', enabled=(device=="cuda")):
                prefix = mapper(eb)                 # [B, n_soft, d_model]
                logits = model.forward_with_prefix(xb, prefix)  # [B, P+T, V]
                T = yb.size(1)                      # comprimento alvo (só os tokens musicais)
                logits = logits[:, -T:, :]          # <-- RECORTE AQUI
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
            if train:
                scaler.scale(loss).backward()
                scaler.step(optim); scaler.update(); optim.zero_grad(set_to_none=True)
            total += loss.item(); steps += 1
        return total / max(1, steps)

    for ep in range(1, args.epochs+1):
        tr = run_epoch(tr_loader, True)
        va = run_epoch(va_loader, False)
        print(f"[{ep}] train={tr:.4f} | val={va:.4f}")
        torch.save({"gpt": gpt.state_dict(), "mapper": mapper.state_dict(), "cfg": vars(cfg), "n_soft": args.n_soft, "dim_text": dim_text},
                   out_dir / f"softprompt_ep{ep:02d}.pt")

if __name__ == "__main__":
    main()
