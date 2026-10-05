"""Dev: learned histogram-kernel features. For every (prefix, candidate artist) count the fit days that contain the artist,
binned by (similarity type, similarity bin, neighbour-size bucket)."""
import numpy as np, scipy.sparse as sp, time
from sklearn.preprocessing import normalize
from solution import Data
from common import *

SIZE_EDGES=np.log([12,30])      # neighbour day length buckets (3)
class LK:
    def __init__(s,D,fit_mask,types=('art_bin','art_log','rel_log'),nb=14):
        s.D=D; s.fit_mask=fit_mask; s.fit_idx=np.where(fit_mask)[0]; s.types=types; s.nb=nb
        s.Q={}; s.F={}; s.edges={}
        fi=s.fit_idx
        def idfd(X,w):
            df=np.asarray((X[fi]>0).sum(0)).ravel(); return sp.diags((np.log((len(fi)+1)/(df+1))**w).astype(np.float32))
        def tf(X,k):
            if k=='bin':
                X=X.copy(); X.data[:]=1; return X
            return X.log1p()
        cfg={'art_bin':(D.C1,D.Ca,'bin',.5),'art_log':(D.C1,D.Ca,'log',1.0),'rel_log':(D.R1,D.Ra,'log',.5)}
        for t in types:
            P1,Pa,k,w=cfg[t]; W=idfd(Pa,w)
            s.F[t]=normalize(tf(Pa,k)@W).tocsr()        # all sessions, whole day (rows for non-fit sessions unused)
            s.Q[t]=normalize(tf(P1,k)@W).tocsr()        # first-half bags
        s.size_bucket=np.digitize(np.log(np.maximum(D.Ca.getnnz(1),1)),SIZE_EDGES)   # distinct-artist count of the day
        s.Tcsc=D.Ca.tocsc()
        # similarity edges: log-spaced
        s.edges={t:np.r_[np.geomspace(0.01,0.6,nb-1)] for t in types}
    def row(s,pre_sessions,cand_artists,own_sessions):
        """pre_sessions: 6 session indices. cand_artists: list of 12 artist ids (6 cands x 2). Returns (6,12,len(types)*nb*3)."""
        D=s.D; out=np.zeros((6,len(cand_artists),len(s.types),s.nb,3),np.float32)
        sess_lists=[]; 
        for a in cand_artists:
            ss=s.Tcsc.indices[s.Tcsc.indptr[a]:s.Tcsc.indptr[a+1]]
            ss=ss[s.fit_mask[ss]]; ss=ss[~np.isin(ss,own_sessions)]; sess_lists.append(ss)
        U=np.unique(np.concatenate(sess_lists)) if sess_lists else np.array([],int)
        if len(U)==0: return out
        pos={x:i for i,x in enumerate(U)}
        for ti,t in enumerate(s.types):
            K=(s.Q[t][pre_sessions]@s.F[t][U].T).toarray()          # (6,|U|)
            bins=np.digitize(K,s.edges[t])                          # 0..nb-1
            for ai,ss in enumerate(sess_lists):
                if len(ss)==0: continue
                ix=np.array([pos[x] for x in ss]); sb=s.size_bucket[ss]
                for sbk in range(3):
                    m=sb==sbk
                    if not m.any(): continue
                    for j in range(6):
                        out[j,ai,ti,:,sbk]=np.bincount(bins[j][ix[m]],minlength=s.nb)[:s.nb]
        return out.reshape(6,len(cand_artists),-1)
