import sys,time; sys.path.insert(0,'dev')
from common import *
import torch
from transformers import AutoTokenizer, AutoModel
torch.set_num_threads(2)
tr,te=load()
name=sys.argv[1]
tok=AutoTokenizer.from_pretrained(name); m=AutoModel.from_pretrained(name).eval()
def emb(texts,bs=64):
    out=[]; t0=time.time()
    idx=np.argsort([len(t) for t in texts])
    with torch.no_grad():
        for i in range(0,len(texts),bs):
            b=[texts[j] for j in idx[i:i+bs]]
            x=tok(b,padding=True,truncation=True,max_length=64,return_tensors='pt')
            h=m(**x).last_hidden_state; mk=x['attention_mask'].unsqueeze(-1).float()
            out.append(((h*mk).sum(1)/mk.sum(1)).numpy())
    E=np.concatenate(out); R=np.zeros_like(E); R[idx]=E; print('embed',time.time()-t0); return R
E=emb(tr.text.tolist()); Et=emb(te.text.tolist())
tag=name.split('/')[-1]; np.save(f'dev/emb_{tag}_tr.npy',E); np.save(f'dev/emb_{tag}_te.npy',Et)
