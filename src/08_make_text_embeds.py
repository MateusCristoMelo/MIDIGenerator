import argparse, json
from pathlib import Path
import pandas as pd
from sentence_transformers import SentenceTransformer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="ex.: data/splits/train_5k.csv")
    ap.add_argument("--out_parquet", required=True, help="ex.: data/text_embeds/train_5k.parquet")
    ap.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    texts = df["caption"].fillna("").astype(str).tolist()
    model = SentenceTransformer(args.model)
    embs = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    out = pd.DataFrame({
        "midi_path": df["midi_path"].astype(str).tolist(),
        "caption": texts,
        "embed": [e.tolist() for e in embs]
    })
    Path(args.out_parquet).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out_parquet, index=False)
    print("Salvo:", args.out_parquet, "| n=", len(out), "| dim=", len(embs[0]))

if __name__ == "__main__":
    main()
