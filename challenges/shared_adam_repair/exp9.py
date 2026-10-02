from exp8 import *
if __name__=="__main__":
    base=dict(cbar=0.0,sc=0.04,s1=0.01,s2=0.0,sig=0.1)
    for ch in [dict(s1=0.001),dict(s1=0.0001),dict(sc=0.1,s1=0.001),dict(sig=0.03,s1=0.001),dict(sig=0.3,s1=0.001)]:
        hp={**base,**ch}; evaluate(mk(hp),idx,str(ch))
