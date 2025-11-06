import argparse
import pandas as pd
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json_path', type=str,
                    default=r'..\data\train.json',
                    help='Caminho local para o train.json do MidiCaps')
    ap.add_argument('--sample_size', type=int, default=25000)
    ap.add_argument('--min_duration', type=float, default=8.0)
    ap.add_argument('--max_duration', type=float, default=360.0)
    # ap.add_argument('--allowed_time_sigs', type=str, default='4/4,3/4')
    ap.add_argument('--midi_root', type=str, default=r'..\data')
    ap.add_argument('--out_csv', type=str, default=r'..\data\splits\train_25k.csv')
    args = ap.parse_args()

    print('Lendo JSON local:', args.json_path)
    df = pd.read_json(args.json_path, lines=True)

    # allowed_ts = set([s.strip() for s in args.allowed_time_sigs.split(',') if s.strip()])
    df = df[df['duration'].between(args.min_duration, args.max_duration)]
    # df = df[df['time_signature'].isin(allowed_ts)]
    df = df[~df['caption'].isna() & (df['caption'].str.len() > 5)]

    midi_root = Path(args.midi_root)
    def resolve_path(loc):
        return str(midi_root / str(loc))

    df['midi_path'] = df['location'].apply(resolve_path)
    df['exists'] = df['midi_path'].apply(lambda p: Path(p).exists())
    df = df[df['exists']].copy()

    df = df.sample(n=min(args.sample_size, len(df)), random_state=42)

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    cols = ['midi_path', 'caption', 'tempo', 'time_signature', 'duration',
            'key', 'instrument_summary']
    # só salva colunas que realmente existem no JSON
    cols = [c for c in cols if c in df.columns]
    df[cols].to_csv(args.out_csv, index=False)

    print('Salvo em:', args.out_csv)

if __name__ == '__main__':
    main()
