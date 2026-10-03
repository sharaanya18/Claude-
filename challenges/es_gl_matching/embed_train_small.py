import json, numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModel
P="dataset/public"; name="intfloat/multilingual-e5-small"
F=[json.loads(l) for l in open(f"{P}/folios.jsonl")]; tr=set(pd.read_csv(f"{P}/train.csv").folio_id); F=[f for f in F if f["folio_id"] in tr]
tok=AutoTokenizer.from_pretrained(name); m=AutoModel.from_pretrained(name).eval(); torch.set_num_threads(4)
texts=[];meta=[]
for fi,f in enumerate(F):
    for c in f["spanish_clues"]: texts.append(c["text"]); meta.append((fi,"es",c["target_id"]))
    for c in f["galician_descriptors"]: texts.append(c["text"]); meta.append((fi,"gl",c["descriptor_id"]))
order=np.argsort([len(t) for t in texts]); E=np.zeros((len(texts),384),np.float32)
for i in range(0,len(texts),64):
    idx=order[i:i+64]; b=tok(["query: "+texts[j] for j in idx],padding=True,truncation=True,max_length=256,return_tensors="pt")
    with torch.no_grad(): h=m(**b).last_hidden_state
    mk=b["attention_mask"][...,None]; e=torch.nn.functional.normalize((h*mk).sum(1)/mk.sum(1),dim=-1); E[idx]=e.numpy()
np.savez("train_e5s.npz",E=E,fi=np.array([x[0] for x in meta]),lang=np.array([x[1] for x in meta]),pid=np.array([x[2] for x in meta]),folio=np.array([F[x[0]]["folio_id"] for x in meta]))
print("done",E.shape)
