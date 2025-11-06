# 07_train_gpt.py
# decoder-only com AMP, grad checkpoint e early-stopping.
# Lê shards .npy do binpack (step 06).

import argparse, json, math, time, csv, os, random
from pathlib import Path
from typing import List, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint

# -------------------------
# Utils
# -------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def list_shards(bin_dir: Path, split: str, seq_len: int) -> List[Path]:
    pat = f"{split}_len{seq_len}_shard"
    return sorted(bin_dir.glob(f"{pat}*.npy"))

def load_meta(bin_dir: Path, split: str) -> dict:
    meta_path = bin_dir / f"{split}_index.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)

# -------------------------
# Dataset
# -------------------------

class BinpackDataset(Dataset):
    def __init__(self, shard_paths: List[Path], mmap=True):
        self.shards = shard_paths
        if not self.shards:
            raise RuntimeError("Nenhum shard encontrado.")
        self.mmaps = []
        self.rows_per_shard = []
        self.seq_len = None
        for p in self.shards:
            # mmap para economizar RAM
            arr = np.load(p, mmap_mode="r" if mmap else None)
            if self.seq_len is None:
                self.seq_len = arr.shape[1]
            self.mmaps.append(arr)
            self.rows_per_shard.append(arr.shape[0])

        # índice (shard_idx, row_idx)
        self.index = []
        for si, nrows in enumerate(self.rows_per_shard):
            for r in range(nrows):
                self.index.append((si, r))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        si, ri = self.index[i]
        arr = self.mmaps[si][ri]  # shape [seq_len] uint32
        # x = seq[:-1], y = seq[1:]
        x = torch.from_numpy(arr[:-1].astype(np.int64))
        y = torch.from_numpy(arr[1:].astype(np.int64))
        return x, y

def make_loader(shards: List[Path], batch_size: int, shuffle: bool, num_workers: int):
    ds = BinpackDataset(shards)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True, drop_last=True)

# -------------------------
# Modelo (pequeno GPT)
# -------------------------

class GPTConfig:
    def __init__(self, vocab_size: int, seq_len: int,
                 n_layers=6, n_heads=8, d_model=512, d_ff=2048,
                 dropout=0.1, grad_checkpoint=True):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        self.grad_checkpoint = grad_checkpoint

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

        # máscara causal
        self.register_buffer("mask", torch.tril(torch.ones(cfg.seq_len, cfg.seq_len)).unsqueeze(0).unsqueeze(0))
        # shape [1,1,T,T]

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)  # [B,T,3C]
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # [B, nh, T, hd]
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B,nh,T,T]
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # [B,nh,T,hd]
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.act = nn.GELU()

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))

class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)
        self.grad_checkpoint = cfg.grad_checkpoint

    def forward(self, x):
        def attn_fn(x_in):
            return self.attn(self.ln1(x_in))
        def mlp_fn(x_in):
            return self.mlp(self.ln2(x_in))

        if self.grad_checkpoint and self.training:
            x = x + checkpoint(attn_fn, x, use_reentrant=False)
            x = x + checkpoint(mlp_fn, x, use_reentrant=False)
        else:
            x = x + self.attn(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        # idx: [B, T]
        B, T = idx.shape
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)  # [1,T]
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)  # [B,T,V]
        return logits

# -------------------------
# Treino
# -------------------------

def train_one_epoch(model, loader, optimizer, scaler, device, pad_id, grad_accum,
                    scheduler=None, use_amp=True, clip_grad=1.0,
                    print_every=50, save_every_steps=0, ckpt_dir=None,
                    state=None, max_steps_per_epoch=0):
    model.train()
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
    running, steps = 0.0, 0
    if state is None: state = {"global_step": 0}

    optimizer.zero_grad(set_to_none=True)

    for it, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', enabled=use_amp):
            logits = model(x)
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            loss = loss / grad_accum

        scaled_loss = torch.amp.GradScaler.get_scaler().scale(loss) if False else None  # só pra IDE não reclamar
        scaled_loss = scaler.scale(loss)
        scaled_loss.backward()

        if (it + 1) % grad_accum == 0:
            if clip_grad and clip_grad > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        state["global_step"] += 1
        if print_every and state["global_step"] % print_every == 0:
            print(f"step {state['global_step']}: loss={loss.item()*grad_accum:.4f}")

        if save_every_steps and state["global_step"] % save_every_steps == 0 and ckpt_dir is not None:
            save_ckpt(Path(ckpt_dir) / f"step_{state['global_step']:07d}.pt",
                      model, optimizer, scheduler, scaler, {}, 0, 0.0, best=False)

        running += loss.item() * grad_accum
        steps += 1

        if max_steps_per_epoch and steps >= max_steps_per_epoch:
            break

    return running / max(1, steps)

@torch.no_grad()
def evaluate(model, loader, device, pad_id):
    model.eval()
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
    running, steps = 0.0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
        running += loss.item()
        steps += 1
    return running / max(1, steps)

def cosine_with_warmup(optimizer, warmup_steps, total_steps, min_lr=1e-6):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def save_ckpt(path: Path, model, optim, sched, scaler, cfg: dict, epoch: int, val_loss: float, best: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optim.state_dict(),
        "scheduler": sched.state_dict() if sched else None,
        "scaler": scaler.state_dict(),
        "cfg": cfg,
        "epoch": epoch,
        "val_loss": val_loss,
        "best": best,
    }, path)

# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin_dir", type=str, required=True, help="pasta do binpack (step 06)")
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=2)  # 1050Ti: 2 é seguro em 1024
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_steps", type=int, default=200)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=0) # Windows: 0 evita dor de cabeça
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--n_layers", type=int, default=6)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--d_ff", type=int, default=2048)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--no_grad_ckpt", action="store_true")
    ap.add_argument("--out_dir", type=str, default="runs/exp1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--early_patience", type=int, default=3)
    ap.add_argument("--max_steps_per_epoch", type=int, default=0,help="Se >0, limita passos por época (útil p/ checkpoint rápido)")
    ap.add_argument("--save_every_steps", type=int, default=0,help="Se >0, salva checkpoint a cada N passos")
    ap.add_argument("--print_every", type=int, default=50,help="Print de progresso a cada N passos")

    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    bin_dir = Path(args.bin_dir)
    # vocabulário
    with open(bin_dir / "vocab.json", "r", encoding="utf-8") as f:
        stoi = json.load(f)
    vocab_size = max(int(v) for v in stoi.values()) + 1

    # pega PAD/BOS/EOS/UNK dos metadados de train (se existirem)
    train_meta = load_meta(bin_dir, "train")
    special = train_meta.get("special_ids", {"PAD": 0, "BOS": 1, "EOS": 2, "UNK": 3})
    pad_id = int(special.get("PAD", 0))

    # shards
    train_shards = list_shards(bin_dir, "train", args.seq_len)
    val_shards   = list_shards(bin_dir, "val",   args.seq_len)
    test_shards  = list_shards(bin_dir, "test",  args.seq_len)
    if not train_shards or not val_shards:
        raise RuntimeError("Faltam shards de train/val para seq_len fornecido.")

    train_loader = make_loader(train_shards, args.batch_size, shuffle=True,  num_workers=args.num_workers)
    val_loader   = make_loader(val_shards,   args.batch_size, shuffle=False, num_workers=args.num_workers)

    # modelo
    cfg = GPTConfig(
        vocab_size=vocab_size,
        seq_len=args.seq_len,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_model=args.d_model,
        d_ff=args.d_ff,
        dropout=args.dropout,
        grad_checkpoint=(not args.no_grad_ckpt),
    )
    model = GPT(cfg).to(device)

    # otimizador / scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    total_steps = args.epochs * (len(train_loader) // max(1, args.grad_accum))
    scheduler = cosine_with_warmup(optimizer, args.warmup_steps, max(args.warmup_steps + 1, total_steps))
    scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda" and not args.no_amp))

    # logging
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.csv"
    if not log_path.exists():
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["epoch","train_loss","val_loss","lr","time_s"])

    best_val = float("inf")
    patience = args.early_patience
    wait = 0

    print(f"Treinando em {device} | vocab={vocab_size} | seq_len={args.seq_len} | bs={args.batch_size} | layers={args.n_layers} | d_model={args.d_model}")

    state = {"global_step": 0}

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
        model, train_loader, optimizer, scaler, device, pad_id, args.grad_accum, scheduler,
        use_amp=(device=="cuda" and not args.no_amp),
        print_every=args.print_every,
        save_every_steps=args.save_every_steps,
        ckpt_dir=out_dir,
        state=state,
        max_steps_per_epoch=args.max_steps_per_epoch,
        )
        
        val_loss = evaluate(model, val_loader, device, pad_id)
        dt = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        with open(log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([ep, f"{train_loss:.4f}", f"{val_loss:.4f}", f"{current_lr:.6f}", f"{dt:.1f}"])

        print(f"[Epoch {ep}] train={train_loss:.4f} | val={val_loss:.4f} | lr={current_lr:.6f} | {dt:.1f}s")

        # checkpoints
        save_ckpt(out_dir / f"last.pt", model, optimizer, scheduler, scaler, vars(args), ep, val_loss, best=False)
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            wait = 0
            save_ckpt(out_dir / f"best.pt", model, optimizer, scheduler, scaler, vars(args), ep, val_loss, best=True)
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping.")
                break

    # avaliação final no test (opcional)
    if test_shards:
        test_loader = make_loader(test_shards, args.batch_size, shuffle=False, num_workers=args.num_workers)
        test_loss = evaluate(model, test_loader, device, pad_id)
        print(f"[TEST] loss={test_loss:.4f} | ppl={math.exp(test_loss):.2f}")

if __name__ == "__main__":
    main()
