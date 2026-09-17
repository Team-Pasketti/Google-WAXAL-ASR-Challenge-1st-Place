#!/usr/bin/env python3
"""Reproduce the complete leaderboard-confirmed V6 postprocessing pipeline.

This is the standalone form of the inline rule-based command originally used
to create `submission_v6_stage2_8_lb_0.7555_pp_best.csv`.
"""
from __future__ import annotations
import argparse,csv,re,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'remote'))
from respace_probe import waxal_tables,apply_split
from form_probe import apply_form

JOIN_RULES=[('ba',x) for x in ['zali','tie','lati','komi','fandi','vandi','sali','telemi','lakisi']]
NA_RULES=[('na',x) for x in ['ye','nga','kati','se','yango','ba','biso']]
RARE_RIKU=re.compile(r'(?i)^(a|va|i|ri|chi|zvi|dzi|u|pa|ti|ndi|ku)ri(ku.+)$')
TRAIL='.,!?;:'

def split_ba(text,table):
 out=[];hits=0
 for word in text.split():
  core=word.strip(TRAIL);trail=word[len(word.rstrip(TRAIL)):];mapped=table.get(core.lower())
  if mapped:
   if core[:1].isupper():mapped=mapped[:1].upper()+mapped[1:]
   out.append(mapped+trail);hits+=1
  else:out.append(word)
 return ' '.join(out),hits

def split_rare_riku(text):
 out=[];hits=0
 for word in text.split():
  core=word.strip(TRAIL);trail=word[len(word.rstrip(TRAIL)):];m=RARE_RIKU.match(core)
  if m:
   k=len(m.group(1))+2;out.append(core[:k]+' '+core[k:]+trail);hits+=1
  else:out.append(word)
 return ' '.join(out),hits

def finish(text):
 text=text.strip()
 if not text:return text
 for i,ch in enumerate(text):
  if ch.isalpha():text=text[:i]+ch.upper()+text[i+1:];break
 if text[-1] not in '.?!':text+='.'
 return text

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--submission',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--pool',type=Path,default=HERE/'corpus/pool');ap.add_argument('--lid',type=Path,default=HERE.parent/'newaudios_announced3_predictions.csv');a=ap.parse_args()
 lr=list(csv.DictReader(a.lid.open(encoding='utf-8-sig')));ic='ID' if 'ID' in lr[0] else 'id';lid={r[ic]:r['announced3_language'] for r in lr}
 lin,_=waxal_tables(a.pool,'lin',3,.95);ba={j.lower():s.lower() for j,s in lin.items() if s.split()[0].lower() in {'ba','bazo','baza'}}
 sna,_=waxal_tables(a.pool,'sna',3,.95,grammatical=True)
 rows=list(csv.DictReader(a.submission.open(encoding='utf-8-sig',newline='')));result=[];counts={}
 def run(key,fn,text):
  text,n=fn(text);counts[key]=counts.get(key,0)+n;return text
 for row in rows:
  text=row['Target'];lang=lid.get(row['ID'])
  if lang=='lin':
   text=run('namoni -> na moni',lambda z:apply_form(z,'na','moni','split'),text)
   text=run('ambiguous ba/bazo/baza split',lambda z:split_ba(z,ba),text)
   text=run('nazomona -> nazo mona',lambda z:apply_form(z,'nazo','mona','split'),text)
   text=run('tozomona -> tozo mona',lambda z:apply_form(z,'tozo','mona','split'),text)
   for x,y in JOIN_RULES:text=run(f'{x} {y} -> {x+y}',lambda z,x=x,y=y:apply_form(z,x,y,'join'),text)
   for x,y in NA_RULES:text=run(f'{x+y} -> {x} {y}',lambda z,x=x,y=y:apply_form(z,x,y,'split'),text)
  elif lang=='sna':
   text=run('Shona frequent grammatical split',lambda z:apply_split(z,sna),text)
   text=run('Shona rare ri+ku split',split_rare_riku,text)
  result.append((row['ID'],finish(text)))
 with a.out.open('w',encoding='utf-8',newline='') as f:w=csv.writer(f);w.writerow(['ID','Target']);w.writerows(result)
 print(f'wrote {a.out}: {len(result)} rows')
 for key,n in counts.items():print(f'  {key}: {n}')
if __name__=='__main__':main()
