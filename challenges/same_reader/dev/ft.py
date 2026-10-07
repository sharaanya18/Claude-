import sys,time,random,os; sys.path.insert(0,'dev')
os.environ["TOKENIZERS_PARALLELISM"]="false"
from common import *
import torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
torch.set_num_threads(4)
name=sys.argv[1]; fold=int(sys.argv[2]); EPOCHS=int(sys.argv[3]); LR=float(sys.argv[4]); NTR=int(sys.argv[5]) if len(sys.argv)>5 else 0
SEED=42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
tr,te=load(); tr=tr.reset_index(drop=True)
EM=sorted(set(tr.emotion_a)|set(tr.emotion_b)|set(te.emotion_a)|set(te.emotion_b)); ei={e:i for i,e in enumerate(EM)}
tok=AutoTokenizer.from_pretrained(name)
def enc(df):
    s=(df.emotion_a+" and "+df.emotion_b+" ("+df.annotator_count.astype(str)+" raters)").tolist()
    x=tok(s,df.text.tolist(),truncation='only_second',max_length=72)
    return x['input_ids'],df.emotion_a.map(ei).values,df.emotion_b.map(ei).values,(df.annotator_count.values-3)
class M(nn.Module):
    def __init__(s):
        super().__init__(); s.b=AutoModel.from_pretrained(name); h=s.b.config.hidden_size
        s.ee=nn.Embedding(len(EM),32); s.ce=nn.Embedding(3,8); s.drop=nn.Dropout(0.1)
        s.head=nn.Sequential(nn.Linear(h+32+8,128),nn.GELU(),nn.Dropout(0.1),nn.Linear(128,1))
    def forward(s,ids,mask,a,b,c):
        h=s.b(input_ids=ids,attention_mask=mask).last_hidden_state; m=mask.unsqueeze(-1).float()
        p=(h*m).sum(1)/m.sum(1)
        e=s.ee(a)+s.ee(b)
        return s.head(torch.cat([s.drop(p),e,s.ce(c)],1)).squeeze(-1)
def batch(X,idx):
    ids=[X[0][i] for i in idx]; L=max(map(len,ids)); pad=tok.pad_token_id
    I=torch.tensor([r+[pad]*(L-len(r)) for r in ids]); Mk=(I!=pad).long() if False else torch.tensor([[1]*len(r)+[0]*(L-len(r)) for r in ids])
    return I,Mk,torch.tensor(X[1][idx]),torch.tensor(X[2][idx]),torch.tensor(X[3][idx])
def predict(m,X,bs=128):
    m.eval(); n=len(X[0]); order=np.argsort([len(r) for r in X[0]]); out=np.zeros(n)
    with torch.no_grad():
        for i in range(0,n,bs):
            idx=order[i:i+bs]; out[idx]=torch.sigmoid(m(*batch(X,idx))).numpy()
    return out
y=tr.p_same_reader.values.astype(int)
a,b=splits(tr,42,5)[fold]
if NTR: a=a[:NTR]
Xa,Xb=enc(tr.iloc[a]),enc(tr.iloc[b]); ya=torch.tensor(y[a]).float()
g=torch.Generator(); g.manual_seed(SEED)
m=M(); opt=torch.optim.AdamW([{'params':m.b.parameters(),'lr':LR},{'params':list(m.ee.parameters())+list(m.ce.parameters())+list(m.head.parameters()),'lr':1e-3}],weight_decay=0.01)
BS=32; steps=EPOCHS*((len(a)+BS-1)//BS); sch=get_linear_schedule_with_warmup(opt,int(0.06*steps),steps)
lossf=nn.BCEWithLogitsLoss(); t0=time.time()
for ep in range(EPOCHS):
    m.train(); perm=torch.randperm(len(a),generator=g).numpy(); tl=0
    for k,i in enumerate(range(0,len(a),BS)):
        idx=perm[i:i+BS]; I,Mk,A,B,C=batch(Xa,idx)
        loss=lossf(m(I,Mk,A,B,C),ya[idx]); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step(); sch.step(); tl+=loss.item()
        if k%50==0: print(f'  ep{ep} step{k} loss{tl/(k+1):.4f} t{time.time()-t0:.0f}',flush=True)
    p=predict(m,Xb); print(f'EPOCH {ep} fold{fold} AP {AP(y[b],p):.4f} t{time.time()-t0:.0f}',flush=True)
    np.save(f"dev/ft2_{name.split(chr(47))[-1]}_f{fold}_e{ep}.npy",p)
