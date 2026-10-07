import sys,time; sys.path.insert(0,'dev')
from common import *
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
torch.set_num_threads(4)
name=sys.argv[1]; tr,te=load()
tok=AutoTokenizer.from_pretrained(name); m=AutoModelForSequenceClassification.from_pretrained(name).eval()
print(m.config.id2label)
def score(texts,emos,bs=64):
    hyp=[f"The writer feels {e}." for e in emos]; order=np.argsort([len(t) for t in texts]); out=np.zeros((len(texts),m.config.num_labels),dtype=np.float32); t0=time.time()
    with torch.no_grad():
        for i in range(0,len(texts),bs):
            idx=order[i:i+bs]
            x=tok([texts[j] for j in idx],[hyp[j] for j in idx],padding=True,truncation=True,max_length=96,return_tensors='pt')
            out[idx]=m(**x).logits.numpy()
    print('t',time.time()-t0,flush=True); return out
tag=name.split('/')[-1]
for nm,d in (('tr',tr),('te',te)):
    A=score(d.text.tolist(),d.emotion_a.tolist()); B=score(d.text.tolist(),d.emotion_b.tolist())
    np.save(f'dev/nli_{tag}_{nm}.npy',np.stack([A,B],1))
