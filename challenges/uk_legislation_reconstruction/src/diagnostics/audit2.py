import pandas as pd, json, re, textwrap
D="dataset/public/"
c=pd.read_csv(D+"corpus.csv",keep_default_na=False).set_index('unit_id')
tr=pd.read_csv(D+"train.csv",keep_default_na=False)
tt=pd.read_csv(D+"train_targets.csv",keep_default_na=False)
tt['ids']=tt.amending_ids.map(json.loads)
m=tr.merge(tt,on='item_id')
for i in [3,10,57]:
    r=m.iloc[i]
    print("="*110)
    print(f"QUERY {r.item_id} | {r.act_title} ({r.act_citation}) {r.section_label} @ {r.date}")
    print("ENACTED:", textwrap.shorten(r.enacted_text,700))
    print("-"*40,"AT DATE:")
    print(textwrap.shorten(r.text_at_date,900))
    print("-"*40,f"GOLD {len(r.ids)} ids:")
    for uid in r.ids:
        g=c.loc[uid]
        print(f"  [{uid}] {g.document_title} | {g.citation} | {g.document_date} | {g.label}")
        print(f"      ctx: {g.context[:160]}")
        print(f"      txt: {textwrap.shorten(g.text,900)}")
