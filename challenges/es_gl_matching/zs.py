import json, numpy as np, pandas as pd, torch, time, sys
from transformers import AutoTokenizer, AutoModel
from scipy.optimize import linear_sum_assignment
P="dataset/public"; name=sys.argv[1]; nf=int(sys.argv[2])
F=[json.loads(l) for l in open(f"{P}/folios.jsonl")]
tt=pd.read_csv(f"{P}/train_targets.csv"); gold={r.target_id:json.loads(r.prediction)["descriptor_id"] for r in tt.itertuples()}
tr=set(pd.read_csv(f"{P}/train.csv").folio_id); F=[f for f in F if f["folio_id"] in tr][:nf]
tok=AutoTokenizer.from_pretrained(name); m=AutoModel.from_pretrained(name).eval(); torch.set_num_threads(4)
def emb(texts,prefix):
    out=[]
    if "Qwen" in name:
        tok.padding_side="left"
        for i in range(0,len(texts),16):
            b=tok([prefix+t for t in texts[i:i+16]],padding=True,truncation=True,max_length=256,return_tensors="pt")
            with torch.no_grad(): h=m(**b).last_hidden_state[:,-1]
            out.append(torch.nn.functional.normalize(h.float(),dim=-1))
        return torch.cat(out).numpy()
    for i in range(0,len(texts),32):
        b=tok([prefix+t for t in texts[i:i+32]],padding=True,truncation=True,max_length=256,return_tensors="pt")
        with torch.no_grad(): h=m(**b).last_hidden_state
        mask=b["attention_mask"][...,None]; e=(h*mask).sum(1)/mask.sum(1); out.append(torch.nn.functional.normalize(e,dim=-1))
    return torch.cat(out).numpy()
t=time.time(); acc_arg=[];acc_h=[]
for f in F:
    q=[c["text"] for c in f["spanish_clues"]]; d=[c["text"] for c in f["galician_descriptors"]]
    pre=("query: ","query: ") if "e5" in name else (("Instruct: Match the Spanish description with the Galician description of the same concept\nQuery:","") if "Qwen" in name else ("",""))
    S=emb(q,pre[0])@emb(d,pre[1]).T
    ids=[c["descriptor_id"] for c in f["galician_descriptors"]]; g=[ids.index(gold[c["target_id"]]) for c in f["spanish_clues"]]
    acc_arg.append(np.mean(S.argmax(1)==np.array(g))); r,c=linear_sum_assignment(-S); acc_h.append(np.mean(c==np.array(g)))
print(name,"folios",len(F),"argmax acc %.3f hungarian acc %.3f"%(np.mean(acc_arg),np.mean(acc_h)),"%.0fs"%(time.time()-t))
