import pandas as pd,re
c=pd.read_csv('dataset/public/corpus.csv',keep_default_na=False)
def lblkey(l):
    m=re.match(r'Sch\. (\d+)(?: para\. (\d+))?',l)
    if m: return (1,int(m.group(1)), int(m.group(2)) if m.group(2) else 0)
    m=re.match(r's\. (\d+)',l)
    if m: return (0,int(m.group(1)),0)
    return (2,0,0)
for cit,ctxsub in [('2005 c. 15','SCHEDULE 2'),('2007 c. 27','SCHEDULE 8'),('2011 c. 13','SCHEDULE 14')]:
    d=c[(c.citation==cit)&(c.context.str.contains(ctxsub,regex=False))].copy()
    d['k']=d.label.map(lblkey)
    print('====',cit,ctxsub,'n=',len(d))
    for r in d.sort_values('k').head(4).itertuples():
        print(' ',r.label,'|ctx:',r.context[:170]); print('    ',r.text[:240].replace('\n',' '))
    # also the section that introduces the schedule
    s=c[(c.citation==cit)&(c.text.str.contains(ctxsub.replace('SCHEDULE','Schedule'),regex=False))&(~c.context.str.contains('SCHEDULE',regex=False))]
    print('  intro sections:',len(s))
    for r in s.head(3).itertuples(): print('   >',r.label,r.text[:200])
