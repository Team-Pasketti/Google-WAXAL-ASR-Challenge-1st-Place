#!/usr/bin/env python3
"""Filter correlated-checkpoint repairs through architecturally diverse members.

The input chain is base -> ghost -> relaxed majority. A ghost substitution is
kept only if more diverse members emit its replacement than its original. A
relaxed-majority substitution is kept only with >=3 diverse votes and a margin
of >=2. All other proposed changes revert to the base. No lexical exception or
row ID is hard-coded.
"""
from __future__ import annotations
import argparse,csv
from collections import Counter
from pathlib import Path
TRAIL='.,!?;:"\'()'
def load(p):return {r['ID']:r['Target'] for r in csv.DictReader(p.open(encoding='utf-8-sig'))}
def core(w):return w.strip(TRAIL).lower()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);ap.add_argument('--ghost',type=Path,required=True);ap.add_argument('--majority',type=Path,required=True);ap.add_argument('--members',type=Path,nargs='+',required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--audit',type=Path,required=True);a=ap.parse_args()
 base,ghost,majority=map(load,[a.base,a.ghost,a.majority]);members=[load(p) for p in a.members];assert set(base)==set(ghost)==set(majority)
 out=[];audit=[]
 for uid,text in base.items():
  bw=text.split();gw=ghost[uid].split();mw=majority[uid].split();assert len(bw)==len(gw)==len(mw),(uid,len(bw),len(gw),len(mw))
  sets=[{core(w) for w in m[uid].split() if core(w).isalpha()} for m in members];final=list(bw)
  for i,(b,g,m) in enumerate(zip(bw,gw,mw)):
   bc,gc,mc=core(b),core(g),core(m)
   if bc!=gc:
    ov=sum(bc in s for s in sets);nv=sum(gc in s for s in sets);keep=nv>ov
    audit.append({'ID':uid,'stage':'ghost','old':bc,'new':gc,'diverse_old':ov,'diverse_new':nv,'kept':int(keep)})
    if keep:final[i]=g
   if gc!=mc:
    ov=sum(gc in s for s in sets);nv=sum(mc in s for s in sets);keep=nv>=3 and nv-ov>=2
    audit.append({'ID':uid,'stage':'nonword_majority','old':gc,'new':mc,'diverse_old':ov,'diverse_new':nv,'kept':int(keep)})
    if keep:final[i]=m
  out.append((uid,' '.join(final)))
 with a.out.open('w',encoding='utf8',newline='') as f:w=csv.writer(f);w.writerow(['ID','Target']);w.writerows(out)
 with a.audit.open('w',encoding='utf8',newline='') as f:
  fields=['ID','stage','old','new','diverse_old','diverse_new','kept'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(audit)
 c=Counter((r['stage'],r['kept']) for r in audit);print(f'wrote {a.out}; decisions={dict(c)}; changed rows={sum(dict(out)[u]!=base[u] for u in base)}')
if __name__=='__main__':main()
