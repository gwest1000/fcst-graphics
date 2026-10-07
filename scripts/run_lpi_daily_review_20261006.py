#!/usr/bin/env python3
"""Frozen-archive daily-primary, rolling-origin LPI review; no operational changes."""
import os, sys, json, datetime as dt, collections, dataclasses, warnings
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1])); sys.path.insert(0,str(Path(__file__).resolve().parent))
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/fcstgraphics-mpl-cache')
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
import run_lpi_review_20261005 as prev
import analyze_lpi_calibration as old
import lightning_ml_archive as archive
SRC=Path('output/lpi_calibration/review_20261005'); OUT=Path('output/lpi_calibration/review_20261006_24h'); OUT.mkdir(parents=True,exist_ok=True)
ST=prev.Study(SRC); N=len(ST.cor); SHAPE=tuple(ST.man['shape']); IX=ST.grid['indices']
DOM={'all_bc':np.ones(N,bool),'corridor':ST.cor}
NAMES=[f'max_sigma{s}' for s in (0,5,10,15,20)]+[f'blend_sigma{s}' for s in (0,5,10,15,20)]
FOLDS={
 'summer':{'train':('2026-07-16','2026-07-31'),'validate':('2026-08-03','2026-08-10'),'test':('2026-08-13','2026-08-24')},
 'late_summer':{'train':('2026-07-16','2026-08-12'),'validate':('2026-08-15','2026-08-24'),'test':('2026-08-27','2026-09-09')},
 'autumn':{'train':('2026-07-16','2026-09-01'),'validate':('2026-09-05','2026-09-17'),'test':('2026-09-21','2026-10-04')}}

def split(day,fold):
 for k,(a,b) in FOLDS[fold].items():
  if a<=day<=b:return k
 return None

def run_ok(c,fold):
 init=prev.parse(c['run']); days=[init.date().isoformat(),(init+dt.timedelta(days=1)).date().isoformat()]
 labels=[split(day,fold) for day in days]
 return labels[0] is not None and labels[0]==labels[1]

def read(r):
 p=Path(r['path']); assert prev.digest(p)==r['sha256']
 with np.load(p) as z:
  assert str(z['cache_grid_hash'].item())==ST.man['grid_hash']
  a=z['potential_x2'].astype(np.float32); a[a==255]=np.nan; return a*.5

def hist(a,y,mask,nb=200,prob=False):
 good=mask & np.isfinite(a) & (y!=255); bins=np.clip((a[good]*(nb if prob else nb/100)).astype(int),0,nb-1)
 return np.stack((np.bincount(bins,minlength=nb),np.bincount(bins,weights=y[good].astype(float),minlength=nb)))

def prepare():
 cache=OUT/'daily_cache.npz'
 if cache.exists():
  data=np.load(cache); cases=json.loads((OUT/'daily_cases.json').read_text()); return data,cases
 byrun=collections.defaultdict(dict)
 for r in ST.records: byrun[r['run']][r['fhour']]=r
 cases=[]; arrays=[]; blocks=[]; targets=[]; coverage=[]
 for run,rs in sorted(byrun.items()):
  if prev.parse(run).hour!=12:continue
  for horizon in (1,2):
   hours=list(range(3+24*(horizon-1),25+24*(horizon-1),3))
   if not all(h in rs for h in hours):continue
   rr=[rs[h] for h in hours]; oi=[ST.obs_by.get(r['valid']) for r in rr]
   if any(i is None for i in oi):continue
   day=rr[0]['day']; arr=np.stack([read(r) for r in rr]); strict=np.isfinite(arr).all(0)
   obs=ST.targets[oi,2]; y=np.max(obs,axis=0); yy=np.where((obs!=255).all(0),y,255).astype('uint8')
   daily=[]; block=[]
   # Keep first day-one block unchanged: F000 is instantaneous, not a preceding 3h maximum.
   preceding=rs.get(hours[0]-3); pre=read(preceding) if preceding is not None and preceding['fhour']>0 else arr[0]
   blend=(arr+np.concatenate((pre[None],arr[:-1]),axis=0))*.5
   for field in (arr,blend):
    daily_raw=np.max(field,axis=0)
    for s in (0,5,10,15,20):
     daily.append(prev.smooth(daily_raw,s).ravel()[IX])
     block.append(np.stack([prev.smooth(a,s).ravel()[IX] for a in field]))
   cases.append({'index':len(cases),'run':run,'horizon':horizon,'day':day,'valid':rr[-1]['valid'],'obs':oi,'records':[r['index'] for r in rr]})
   arrays.append(daily); blocks.append(block); targets.append(yy); coverage.append(float(strict.ravel()[IX].mean()))
   if len(cases)%20==0:prev.log(f'Daily preparation {len(cases)}')
 # Pair both horizons on same date and use common support even across horizons.
 pairs=collections.defaultdict(dict)
 for c in cases:pairs[c['day']][c['horizon']]=c['index']
 keep=[i for p in pairs.values() if len(p)==2 for i in p.values()]
 cases=[{**cases[i],'index':j} for j,i in enumerate(keep)]
 arrays=np.asarray(arrays,dtype='float16')[keep]; blocks=np.asarray(blocks,dtype='float16')[keep]; targets=np.asarray(targets)[keep]
 pairs=collections.defaultdict(list)
 for c in cases:pairs[c['day']].append(c['index'])
 masks=np.zeros((len(cases),N),bool)
 for inds in pairs.values():
  common=np.isfinite(arrays[inds]).all(axis=(0,1)) & (targets[inds]!=255).all(0)
  masks[inds]=common
 bh=np.zeros((len(cases),2,10,2,200),float); dh=bh.copy()
 for c in cases:
  i=c['index']; obs=ST.targets[c['obs'],2]
  for d,mask in enumerate(DOM.values()):
   use=masks[i]&mask
   for k in range(10):
    dh[i,d,k]=hist(arrays[i,k],targets[i],use)
    for j in range(8):bh[i,d,k]+=hist(blocks[i,k,j],obs[j],use)/8
 np.savez_compressed(cache,dh=dh,bh=bh,score=arrays,target=targets,mask=masks,coverage=np.array(coverage)[keep])
 prev.write_json(OUT/'daily_cases.json',cases); return np.load(cache),cases

METRICS=[]; CIs=[]; SELECT=[]; INVENTORY=[]
def evaluate(data,cases,names,experiment,reference,paired=True):
 results={}
 for fold in FOLDS:
  # Entire issued run must stay in one split, including its day-one and day-two windows.
  run_splits=collections.defaultdict(set)
  for c in cases: run_splits[c['run']].add(split(c['day'],fold))
  eligible=[c for c in cases if split(c['day'],fold) is not None and run_ok(c,fold)]
  if paired:
   present=collections.defaultdict(set)
   for c in eligible: present[c['day']].add(c['horizon'])
   eligible=[c for c in eligible if len(present[c['day']])==2]
  for h in sorted(set(c['horizon'] for c in cases)):
   for d,domain in enumerate(DOM):
    ids={s:[c['index'] for c in eligible if c['horizon']==h and split(c['day'],fold)==s] for s in ('train','validate','test')}
    if any(not ii for ii in ids.values()):continue
    for s,ii in ids.items():
     z=data['dh'][ii,d,0]; days=[cases[i]['day'] for i in ii]
     INVENTORY.append(dict(experiment=experiment,fold=fold,horizon=h,domain=domain,split=s,days=len(ii),first=min(days),last=max(days),active_days=int(sum(v[1].sum()>0 for v in z)),mean_eligible_cells=float(np.mean(z[:,0].sum(axis=1))),static_cells=int(DOM[domain].sum())))
    for k in sorted(range(len(names)),key=lambda k:names[k]!=reference):
     name=names[k]
     cal={dur:old.pava_probabilities(old.Histogram(*data[key][ids['train'],d,k].sum(0))) for dur,key in [('24h','dh'),('3h','bh')]}
     for dur,key in [('24h','dh'),('3h','bh')]:
      for s,ii in ids.items():
       row=dict(experiment=experiment,fold=fold,horizon=h,domain=domain,duration=dur,split=s,candidate=name,days=len(ii),**prev.histmetrics(data[key][ii,d,k].sum(0),cal[dur]))
       METRICS.append(row); results[fold,h,domain,dur,s,name]=row
     if name!=reference:
      rng=np.random.default_rng(20261006); kk=names.index(reference)
      for dur,key in [('24h','dh'),('3h','bh')]:
       ch=data[key][ids['test'],d,k]; rh=data[key][ids['test'],d,kk]
       rc=old.pava_probabilities(old.Histogram(*data[key][ids['train'],d,kk].sum(0)))
       # Cluster all cells/blocks from each verifying day; circular consecutive 3-day resampling.
       losses=lambda hist,p:(hist[:,1]*(1-p)**2+(hist[:,0]-hist[:,1])*p*p).sum(1)
       cl=losses(ch,cal[dur]); rl=losses(rh,rc); nn=ch[:,0].sum(1); n=len(nn)
       vals={}
       for blocklen in (1,3):
        start=rng.integers(0,n,size=(2000,int(np.ceil(n/blocklen))))
        bi=((start[:,:,None]+np.arange(blocklen))%n).reshape(2000,-1)[:,:n]
        den=nn[bi].sum(1); delta=np.divide((cl[bi]-rl[bi]).sum(1),den,out=np.full(2000,np.nan),where=den>0)
        lo,hi=np.nanquantile(delta,[.025,.975]); vals[blocklen]=(float(lo),float(hi))
       cand=results[fold,h,domain,dur,'test',name]; ref=results[fold,h,domain,dur,'test',reference]
       CIs.append(dict(experiment=experiment,fold=fold,horizon=h,domain=domain,duration=dur,candidate=name,reference=reference,delta_brier=cand['brier']-ref['brier'],delta_auc=cand['auc']-ref['auc'],ci_day_low=vals[1][0],ci_day_high=vals[1][1],ci_3day_low=vals[3][0],ci_3day_high=vals[3][1]))
    ref=results[fold,h,domain,'24h','validate',reference]
    good=[name for name in names if results[fold,h,domain,'24h','validate',name]['auc']>=ref['auc']-.02]
    best=min(good,key=lambda name:results[fold,h,domain,'24h','validate',name]['brier'])
    for tol in (.005,.01,.02):
     close=[name for name in good if results[fold,h,domain,'24h','validate',name]['brier']<=results[fold,h,domain,'24h','validate',best]['brier']*(1+tol)]
     tie=min(close,key=lambda name:results[fold,h,domain,'3h','validate',name]['brier'])
     gain=1-results[fold,h,domain,'3h','validate',tie]['brier']/results[fold,h,domain,'3h','validate',best]['brier']
     chosen=tie if gain>=.02 else best
     SELECT.append(dict(experiment=experiment,fold=fold,horizon=h,domain=domain,daily_tolerance=tol,best_daily=best,chosen=chosen,close_candidates=';'.join(close),brier24_validate=results[fold,h,domain,'24h','validate',chosen]['brier'],brier3_validate=results[fold,h,domain,'3h','validate',chosen]['brier'],relative_3h_gain_vs_daily_best=1-results[fold,h,domain,'3h','validate',chosen]['brier']/results[fold,h,domain,'3h','validate',best]['brier']))
 return results

def ingredients(cases):
 path=OUT/'ingredients_cache.npz'; cp=OUT/'ingredient_cases.json'; formulas=old.candidate_formulas()
 prev.write_json(OUT/'formula_definitions.json',[dataclasses.asdict(f) for f in formulas])
 if path.exists():return np.load(path),json.loads(cp.read_text()),[f.name for f in formulas]
 root=Path('/Volumes/Greg1_2tb/project-data/fcstGraphics/data/lightning_ml'); dd=[]; bb=[]; cc=[]; sources=[]
 for c in cases:
  if c['horizon']!=1:continue
  rd=archive.hourly_lpi_archive_dir(root)/'2026'/c['run']; paths=[rd/f'f{h:03}.npz' for h in range(1,25)]
  if not all(p.exists() for p in paths):continue
  blocks=[]; complete=np.ones(N,bool)
  for j,p in enumerate(paths):
   side=json.loads(p.with_suffix('.json').read_text()); sha=prev.digest(p); assert sha==side['sha256']
   with np.load(p) as z:
    assert str(z['grid_hash'].item())==ST.man['grid_hash']; assert str(z['formula_version'].item())==prev.VERSION
   fields=old.read_hour_fields(p,IX); complete &= np.logical_and.reduce([np.isfinite(v) for key,v in fields.items() if key!='potential']); blocks.append(np.stack([old.compute_formula(fields,f) for f in formulas])); sources.append(dict(path=str(p),sha256=sha))
  block=np.stack(blocks).reshape(8,3,len(formulas),N).max(1).transpose(1,0,2)
  daily=block.max(1); obs=ST.targets[c['obs'],2]; target=np.max(obs,axis=0)
  good=complete & np.isfinite(daily).all(0)&(obs!=255).all(0)
  dh=np.zeros((2,len(formulas),2,200)); bh=dh.copy()
  for d,mask in enumerate(DOM.values()):
   for k in range(len(formulas)):
    dh[d,k]=hist(daily[k],target,good&mask)
    for j in range(8):bh[d,k]+=hist(block[k,j],obs[j],good&mask)/8
  dd.append(dh);bb.append(bh);cc.append({**c,'index':len(cc)})
  if len(cc)%10==0:prev.log(f'Ingredient full days {len(cc)}')
 np.savez_compressed(path,dh=np.array(dd),bh=np.array(bb));prev.write_json(cp,cc);prev.write_csv(OUT/'ingredient_sources.csv',sources)
 return np.load(path),cc,[f.name for f in formulas]

def solar(data,cases):
 # Daily maximum 10 km LPI, compared with logistic LPI-only calibration.
 names=['lpi_only','lpi_daylength','lpi_declination','lpi_calendar']; rows=[]; cis=[]; coef=[]
 score=data['score'][:,2]; mask=data['mask']; targets=data['target']
 for fold in FOLDS:
  runsp=collections.defaultdict(set)
  for c in cases:runsp[c['run']].add(split(c['day'],fold))
  for h in (1,2):
   for domain,dm in DOM.items():
    eligible=[c for c in cases if run_ok(c,fold)]; present=collections.defaultdict(set)
    for c in eligible: present[c['day']].add(c['horizon'])
    eligible=[c for c in eligible if len(present[c['day']])==2]
    ids={s:[c['index'] for c in eligible if c['horizon']==h and split(c['day'],fold)==s] for s in ('train','validate','test')}
    if any(not x for x in ids.values()):continue
    dh=data['dh']; d=list(DOM).index(domain); cal=old.pava_probabilities(old.Histogram(*dh[ids['train'],d,2].sum(0)))
    designs={}; yy={}
    for s,ii in ids.items():
     for i in ii:
      c=cases[i]; use=mask[i]&dm; bins=np.clip((score[i,use]*2).astype(int),0,199); solar=ST.solar(c['valid']); base=logit(np.clip(cal[bins],1e-4,1-1e-4))
      designs[i]=np.column_stack((np.ones(use.sum()),base,(solar['daylength'][use]-14)/4,np.full(use.sum(),solar['declination']/.4),np.full(use.sum(),solar['calendar'])))
      yy[i]=targets[i,use].astype(float)
    per={}
    for name,extra in zip(names,(None,2,3,4)):
     cols=[0,1]+([] if extra is None else [extra]); X=np.concatenate([designs[i][:,cols] for i in ids['train']]); Y=np.concatenate([yy[i] for i in ids['train']]); scale=len(Y)
     def fun(beta):
      z=X@beta; p=expit(z); ridge=.0002*np.sum(beta[1:]**2)
      return (np.sum(np.logaddexp(0,z)-Y*z)/scale+ridge, X.T@(p-Y)/scale+np.r_[0,.0004*beta[1:]])
     fit=minimize(fun,np.r_[-.1,1.,np.zeros(len(cols)-2)],jac=True,method='L-BFGS-B',bounds=[(None,None),(0,None)]+[(None,None)]*(len(cols)-2),options={'maxiter':150});assert fit.success,fit.message
     coef.append(dict(fold=fold,horizon=h,domain=domain,model=name,coefficients=fit.x.tolist(),success=bool(fit.success)))
     for s,ii in ids.items():
      hh=[];loss=[];nn=[]
      for i in ii:
       p=expit(designs[i][:,cols]@fit.x); y=yy[i]; ph=hist(p,y,np.ones(len(y),bool),1000,True);hh.append(ph);loss.append(float(np.sum((p-y)**2)));nn.append(len(y))
      metrics=prev.histmetrics(np.sum(hh,axis=0),(np.arange(1000)+.5)/1000);metrics['brier']=sum(loss)/sum(nn)
      rows.append(dict(experiment='daily_solar',fold=fold,horizon=h,domain=domain,duration='24h',split=s,candidate=name,days=len(ii),**metrics))
      if s=='test':per[name]=(np.array(loss),np.array(nn))
    rng=np.random.default_rng(20261006)
    for name in names[1:]:
     delta=per[name][0]-per['lpi_only'][0];nn=per[name][1];n=len(nn);q={}
     for bl in (1,3):
      bi=((rng.integers(0,n,(2000,int(np.ceil(n/bl))))[:,:,None]+np.arange(bl))%n).reshape(2000,-1)[:,:n];q[bl]=np.quantile(delta[bi].sum(1)/nn[bi].sum(1),[.025,.975])
     a=next(r for r in rows if r['fold']==fold and r['horizon']==h and r['domain']==domain and r['split']=='test' and r['candidate']==name);b=next(r for r in rows if r['fold']==fold and r['horizon']==h and r['domain']==domain and r['split']=='test' and r['candidate']=='lpi_only')
     cis.append(dict(experiment='daily_solar',fold=fold,horizon=h,domain=domain,duration='24h',candidate=name,reference='lpi_only',delta_brier=a['brier']-b['brier'],delta_auc=a['auc']-b['auc'],ci_day_low=q[1][0],ci_day_high=q[1][1],ci_3day_low=q[3][0],ci_3day_high=q[3][1]))
    prev.log(f'Solar daily {fold} day{h} {domain}')
 METRICS.extend(rows);CIs.extend(cis);prev.write_json(OUT/'solar_coefficients.json',coef)

def available_input(data,cases):
 path=OUT/'available_input_cache.npz'
 if path.exists():return np.load(path)
 scores=[]
 for c in cases:
  arr=np.stack([read(ST.records[i]) for i in c['records']]);first=ST.records[c['records'][0]];prior=ST.rec_by.get((c['run'],first['fhour']-3));pre=read(prior) if prior and prior['fhour']>0 else arr[0]
  blend=(arr+np.concatenate((pre[None],arr[:-1]),axis=0))*.5
  vals=[]
  for field in (arr,blend):
   with warnings.catch_warnings():
    warnings.simplefilter('ignore',RuntimeWarning);maximum=np.nanmax(field,axis=0)
   vals.extend([prev.smooth(maximum,km).ravel()[IX] for km in (0,5,10,15,20)])
  scores.append(vals)
 scores=np.array(scores,dtype='float16');pairs=collections.defaultdict(list)
 for c in cases:pairs[c['day']].append(c['index'])
 dh=np.zeros_like(data['dh']);masks=np.zeros((len(cases),N),bool)
 for ids in pairs.values():
  common=np.isfinite(scores[ids]).all((0,1)) & (data['target'][ids]!=255).all(0);masks[ids]=common
  for i in ids:
   for d,mask in enumerate(DOM.values()):
    for k in range(10):dh[i,d,k]=hist(scores[i,k],data['target'][i],common&mask)
 # Companion three-hour values remain on the complete-input support; diagnostic only.
 np.savez_compressed(path,dh=dh,bh=data['bh'],mask=masks,score=scores)
 return np.load(path)

def main():
 data,cases=prepare();prev.log(f'{len(cases)} paired complete daily forecasts');evaluate(data,cases,NAMES,'daily_spatial_timing','max_sigma10')
 available=available_input(data,cases);evaluate(available,cases,NAMES,'daily_available_input','max_sigma10')
 ing,ic,inames=ingredients(cases);prev.log(f'{len(ic)} ingredient days');evaluate(ing,ic,inames,'daily_ingredients','recomputed_baseline',False)
 solar(data,cases)
 prev.write_csv(OUT/'metrics.csv',METRICS);prev.write_csv(OUT/'bootstrap_differences.csv',CIs);prev.write_csv(OUT/'selection.csv',SELECT);prev.write_csv(OUT/'evaluation_inventory.csv',INVENTORY)
 prev.write_json(OUT/'protocol.json',dict(freeze=ST.man['cutoff'],folds=FOLDS,primary='24h all_bc',secondary='3h; corridor independently calibrated',close_daily_tolerances=[.005,.01,.02],material_3h_relative_brier_gain=.02,reference='daily temporal maximum then Gaussian sigma10 km',target='any lightning within30 km during all8 complete3h observation blocks',same_valid_day_horizon_pair=True,common_support='all candidate scores finite; both horizons; complete24h forecast and observations',boundary_rule='Exclude whole48h run when its two daily windows cross splits or buffers; retain same verifying dates for both horizons',bootstraps=2000,seed=20261006,annual_seasonality_identifiable=False,previously_inspected_data=True))
 prev.log('DAILY REVIEW COMPLETE')
if __name__=='__main__':main()
