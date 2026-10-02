import sys, torch, numpy as np, pandas as pd
sys.path.insert(0, "."); import ap_model as M
torch.set_num_threads(4)
train = M.load_rows(pd.read_csv("dataset/public/train.csv", keep_default_na=False))
test = M.load_rows(pd.read_csv("dataset/public/test.csv", keep_default_na=False))
sample = pd.read_csv("dataset/public/sample_submission.csv", keep_default_na=False)
variants = {"C_struct_only": dict(use_tower=False, rot_prob=0.0)}
for name, kw in variants.items():
    logits = None
    for s in range(3):
        m = M.train_model(train, epochs=40, gain_max=1.6, seed=7 + s, log=lambda *a: None, **kw)
        l = M.predict_logits(m, test).numpy(); logits = l if logits is None else logits + l
    logits /= 3
    seqs = []
    for l, r in zip(logits, test):
        pred, _ = M.decode_row(l, r); seqs.append(" ".join(r["ak"][j] for j in pred))
    by = dict(zip([r["id"] for r in test], seqs))
    sub = pd.DataFrame({"example_id": sample.example_id, "target_sequence": [by[i] for i in sample.example_id], "confidence": 0.35})
    sub.to_csv(f"probes/{name}.csv", index=False); print(name, "written", sub.shape, flush=True)
