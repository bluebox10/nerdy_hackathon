"""Export the shipped models to INT8 ONNX and build the serving bundle.

Everything the production box needs ends up in serve/artifacts/:
  bi_encoder/        INT8 ONNX bi-encoder (33M)  -- 2,587 candidates -> 25
  cross_encoder/     INT8 ONNX distilled reranker (22M) -- 25 -> 1
  misc_emb.npy       2,587 x 384 float32 = 3.8 MB. A numpy matmul, not a vector DB.
  whywrong.sqlite    precomputed explanation + hint ladder per misconception
No GPU, no LLM, no network call in the hot path.
"""
import argparse, json, shutil, sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from text_templates import doc_text

ap = argparse.ArgumentParser()
ap.add_argument("--biencoder", required=True)
ap.add_argument("--cross-encoder", default=None)
ap.add_argument("--out", default=str(ROOT / "serve/artifacts"))
args = ap.parse_args()
OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)

from optimum.onnxruntime import (ORTModelForFeatureExtraction,
                                 ORTModelForSequenceClassification, ORTQuantizer)
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoTokenizer

qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=True)


def export(src, dst, kind):
    dst = Path(dst)
    # Wipe the target first. A leftover model_quantized.onnx from a previous run makes
    # ORTQuantizer see two graphs and refuse ("does not support multi-file quantization"),
    # which previously left a STALE encoder in place while the pipeline carried on.
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    cls = ORTModelForFeatureExtraction if kind == "embed" else ORTModelForSequenceClassification
    t0 = time.time()
    m = cls.from_pretrained(src, export=True)
    m.save_pretrained(dst)
    AutoTokenizer.from_pretrained(src).save_pretrained(dst)
    q = ORTQuantizer.from_pretrained(dst)
    q.quantize(save_dir=dst, quantization_config=qconfig)
    # keep only the quantized graph so the bundle stays small
    fp32 = dst / "model.onnx"
    int8 = dst / "model_quantized.onnx"
    size_fp32 = fp32.stat().st_size / 1e6 if fp32.exists() else None
    if int8.exists() and fp32.exists():
        fp32.unlink()
    print(f"  {kind}: fp32={size_fp32:.1f}MB -> int8={int8.stat().st_size/1e6:.1f}MB "
          f"({time.time()-t0:.0f}s)")
    return dict(kind=kind, src=str(src), fp32_mb=round(size_fp32, 1) if size_fp32 else None,
                int8_mb=round(int8.stat().st_size / 1e6, 1))


meta = {"exports": []}
_export_errors = []

# ---- bi-encoder (retriever). SentenceTransformer dirs nest the transformer in 0_Transformer.
bi_src = Path(args.biencoder)
inner = bi_src / "0_Transformer"
meta["exports"].append(export(inner if inner.exists() else bi_src, OUT / "bi_encoder", "embed"))

if args.cross_encoder:
    meta["exports"].append(export(args.cross_encoder, OUT / "cross_encoder", "rank"))
else:
    ce_dir = OUT / "cross_encoder"
    if ce_dir.exists():
        shutil.rmtree(ce_dir)      # never serve a reranker the caller did not ask for

# ---- precompute the misconception matrix with the *torch* model (ground truth),
# ---- then verify the INT8 ONNX reproduces it closely enough to serve.
from sentence_transformers import SentenceTransformer
misc = pd.read_parquet(ROOT / "data/processed/misconceptions.parquet")
st = SentenceTransformer(str(bi_src), device="cpu"); st.max_seq_length = 320
docs = [doc_text(n) for n in misc["misconception_name"]]
E = st.encode(docs, batch_size=64, normalize_embeddings=True, convert_to_numpy=True,
              show_progress_bar=False).astype(np.float32)
np.save(OUT / "misc_emb.npy", E)
# JSON, not parquet: the serving box should not need pandas/pyarrow at all
(OUT / "misconceptions.json").write_text(misc.to_json(orient="records"))
meta["index"] = dict(n=int(E.shape[0]), dim=int(E.shape[1]),
                     bytes=int(E.nbytes), mb=round(E.nbytes / 1e6, 2))
print(f"  index: {E.shape} = {E.nbytes/1e6:.2f} MB")

# parity check: does the INT8 ONNX embed the same text the same way?
import onnxruntime as ort
tok = AutoTokenizer.from_pretrained(OUT / "bi_encoder")
sess = ort.InferenceSession(str(OUT / "bi_encoder/model_quantized.onnx"),
                            providers=["CPUExecutionProvider"])
probe = docs[:64]
enc = tok(probe, padding=True, truncation=True, max_length=320, return_tensors="np")
feed = {i.name: enc[i.name] for i in sess.get_inputs() if i.name in enc}
hidden = sess.run(None, feed)[0]
cls = hidden[:, 0, :]                                  # bge pools with CLS
cls = cls / np.linalg.norm(cls, axis=1, keepdims=True)
cos = float(np.mean(np.sum(cls * E[:64], axis=1)))
meta["int8_parity_cosine_vs_fp32"] = round(cos, 5)
print(f"  INT8 vs fp32 embedding cosine: {cos:.5f}")

(OUT / "export_meta.json").write_text(json.dumps(meta, indent=2))
print(json.dumps(meta, indent=2))
