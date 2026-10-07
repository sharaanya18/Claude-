import sys,pickle,json,numpy as np,pandas as pd
sys.path.insert(0,'src'); import index as IX
sys.modules['__main__'].Corpus=IX.Corpus
X=np.load('working/Xret.npy'); y=np.load('working/yret.npy'); qi=np.load('working/qiret.npy'); cu=np.load('working/curet.npy')
FE=pickle.load(open('working/FE.pkl','rb'))
d=X[:,FE.index('days')]; na=X[:,FE.index('p_napp')]; ni=X[:,FE.index('p_nins')]
print("candidates: %d  gold %d"%(len(y),y.sum()))
for name,m in [("made before date (days>=0)",d>=0),("made after date",d<0)]:
    print(" %-26s n=%5d  gold=%4d  prec=%.3f"%(name,m.sum(),y[m].sum(),y[m].mean()))
m=(d>=0)&(na>0)
print(" before & parser applies an edit : n=%5d gold=%4d prec=%.3f"%(m.sum(),y[m].sum(),y[m].mean()))
m=(d>=0)&(ni>0)
print(" before & parser finds instr     : n=%5d gold=%4d prec=%.3f"%(m.sum(),y[m].sum(),y[m].mean()))
m=(d<0)&(na>0)
print(" AFTER  & parser applies an edit : n=%5d gold=%4d prec=%.3f"%(m.sum(),y[m].sum(),y[m].mean()))
# how far in the past are gold vs non-gold with edits?
g=(y==1)&(d>=0); b=(y==0)&(d>=0)&(na>0)
print("days-before: gold median %.0f  p90 %.0f | nongold-with-edit median %.0f p90 %.0f"%(
  np.median(d[g]),np.percentile(d[g],90),np.median(d[b]),np.percentile(d[b],90)))
