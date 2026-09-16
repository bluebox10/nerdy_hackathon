"""Merge sharded teacher-score files into one distillation dataset."""
import sys, glob
from pathlib import Path
import numpy as np, pandas as pd

out = sys.argv[1]
parts = sorted(glob.glob(sys.argv[2]))
Zs = [np.load(p, allow_pickle=True) for p in parts]
qs = [pd.read_parquet(p.replace(".npz", "_queries.parquet")) for p in parts]
np.savez_compressed(out,
                    cands=np.concatenate([z["cands"] for z in Zs]),
                    scores=np.concatenate([z["scores"] for z in Zs]),
                    gold=np.concatenate([z["gold"] for z in Zs]),
                    src=np.concatenate([z["src"] for z in Zs]))
pd.concat(qs, ignore_index=True).to_parquet(out.replace(".npz", "_queries.parquet"), index=False)
n = sum(len(z["cands"]) for z in Zs)
print(f"merged {len(parts)} shards -> {out}  ({n} groups)")
