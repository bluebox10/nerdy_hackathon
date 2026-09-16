"""Build the canonical WhyWrong dataset + frozen split from the Eedi IR mirror.

Outputs data/processed/{misconceptions,train,val,test}.parquet and split_manifest.json.
The split is frozen here and never re-derived: every later script reads these files.
"""
import json, re, hashlib
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data/raw", ROOT / "data/processed"
OUT.mkdir(parents=True, exist_ok=True)

corpus = pd.read_parquet(RAW / "eedi-ir/corpus/train-00000-of-00001.parquet")
queries = pd.read_parquet(RAW / "eedi-ir/queries/train-00000-of-00001.parquet")
qrels = pd.read_parquet(RAW / "eedi-ir/qrels/train-00000-of-00001.parquet")

misc = corpus.rename(columns={"id_": "misconception_id", "text": "misconception_name"})
misc = misc.sort_values("misconception_id").reset_index(drop=True)

# queries text is "### Question: {subject} | {construct} | {stem}\n### Correct: {c}\n### Incorrect: {d}"
PAT = re.compile(
    r"^### Question:\s*(?P<head>.*?)\n### Correct:\s*(?P<correct>.*?)\n### Incorrect:\s*(?P<incorrect>.*)$",
    re.S,
)

def parse(text):
    m = PAT.match(text.strip())
    if not m:
        return dict(subject="", construct="", question="", correct="", incorrect="")
    head = m.group("head")
    parts = [p.strip() for p in head.split("|")]
    subject = parts[0] if len(parts) > 0 else ""
    construct = parts[1] if len(parts) > 1 else ""
    question = " | ".join(parts[2:]).strip() if len(parts) > 2 else ""
    return dict(subject=subject, construct=construct, question=question,
                correct=m.group("correct").strip(), incorrect=m.group("incorrect").strip())

parsed = pd.DataFrame([parse(t) for t in queries["text"]])
df = pd.concat([queries.reset_index(drop=True), parsed], axis=1)
df = df.rename(columns={"id_": "qid"})
df["question_id"] = df["qid"].str.split("_").str[0].astype(int)
df["answer_letter"] = df["qid"].str.split("_").str[1]
df = df.merge(qrels[["qid", "mid"]], on="qid", how="left").rename(columns={"mid": "misconception_id"})
assert df["misconception_id"].notna().all(), "unlabelled query survived the merge"
df["misconception_id"] = df["misconception_id"].astype(int)
n_bad = (df["question"] == "").sum()
print(f"parse failures: {n_bad}/{len(df)}")

# ---- frozen split: group by question_id so distractors of one question never straddle folds
# folds come from the source mirror and are already question-grouped; verify that.
straddle = df.groupby("question_id")["fold"].nunique().gt(1).sum()
print(f"questions straddling folds: {straddle}")
if straddle:  # fall back to a deterministic hash split on question_id
    df["fold"] = df["question_id"].map(
        lambda q: int(hashlib.md5(f"whywrong-{q}".encode()).hexdigest(), 16) % 5)
    print("-> re-derived folds by deterministic question_id hash")

test = df[df.fold == 0].copy()
val = df[df.fold == 1].copy()
train = df[df.fold.isin([2, 3, 4])].copy()

seen = set(train["misconception_id"])
for name, part in [("val", val), ("test", test)]:
    part["unseen_in_train"] = ~part["misconception_id"].isin(seen)

manifest = dict(
    source="cdtmc/eedi-ir (HF mirror of Eedi 'Mining Misconceptions in Mathematics', NeurIPS 2024)",
    n_misconceptions=int(len(misc)),
    n_queries=int(len(df)),
    n_train=int(len(train)), n_val=int(len(val)), n_test=int(len(test)),
    n_unique_misconceptions_in_train=len(seen),
    n_misconceptions_never_in_train=int(len(misc) - len(seen)),
    test_unseen_queries=int(test["unseen_in_train"].sum()),
    val_unseen_queries=int(val["unseen_in_train"].sum()),
    split_rule="fold 0 = test, fold 1 = val, folds 2-4 = train; grouped by question_id",
)
misc.to_parquet(OUT / "misconceptions.parquet", index=False)
train.to_parquet(OUT / "train.parquet", index=False)
val.to_parquet(OUT / "val.parquet", index=False)
test.to_parquet(OUT / "test.parquet", index=False)
(OUT / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
