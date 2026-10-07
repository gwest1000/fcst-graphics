#!/usr/bin/env python3
"""Creative daily LPI recipes, validation-only refinement, then retrospective tests."""
import sys,os,json,collections,dataclasses,hashlib,warnings,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
import run_lpi_daily_review_20261006 as daily
p=daily.prev;old=daily.old
OUT=Path('output/lpi_calibration/review_20261006_radical');OUT.mkdir(parents=True,exist_ok=True)
INP=OUT/'inputs';INP.mkdir(exist_ok=True)
CASES=json.loads((daily.OUT/'ingredient_cases.json').read_text());N=daily.N;SHAPE=daily.SHAPE;IX=daily.IX;NF=int(np.prod(SHAPE));IDS=np.arange(NF)
KEYS=['mu_li','cape','charge_rh','charge_depth','mid_rh','upward_w','precip_3h','precip_rate','surface_rh','subcloud_rh']
FEATURES=['li_peak','cape_peak','charge_rh_peak','charge_depth_peak','mid_rh_peak','ascent_peak','rain_peak','li_mean','charge_mean','ascent_mean','unstable_fraction','ascent_fraction','instability_charge_peak','charged_ascent_peak','convective_rain_peak','cape_ascent_peak','dry_ascent_peak','humid_surface_peak','humid_subcloud_peak','li_ascent_peak']
MODEL={'additive':list(range(20)),'signed_additive':list(range(20)),'storm_routes':[12,13,14,15,16,7,8,9]}
SEED=20261006

def log(x):p.log(x)
def jwrite(name,x):p.write_json(OUT/name,x)
def coeff_key(fold,name):return fold+'|'+name

def base_configs():
 configs=[dict(name='baseline',family='baseline',scale=20),dict(name='baseline_unsmoothed',family='baseline',scale=0)]
 families=['additive','soft_gate','geometric','bottleneck','noisy_or','ascent_first','instability_first','dry_elevated','stratiform_veto','lagged_ingredients','daily_envelope','storm_dose','persistent_peak']
 for family in families:
  for variant in (0,1):
   configs.append(dict(name=f'{family}_{variant}',family=family,scale=20,a=(.15,.40)[variant],floor=(.10,.30)[variant],uweight=(.40,.70)[variant],power=(.5,1.5)[variant],dose=(.25,.55)[variant],lag=(2,5)[variant]))
 # Unsmoothed diagnostics isolate recipe information from the already-known smoothing gain.
 for family in ('additive','noisy_or','ascent_first','daily_envelope','storm_dose'):
  configs.append(dict(name=family+'_unsmoothed',family=family,scale=0,a=.25,floor=.2,uweight=.55,power=1.,dose=.4,lag=3))
 return configs

def prepare_inputs():
 sources=[]
 root=Path('/Volumes/Greg1_2tb/project-data/fcstGraphics/data/lightning_ml')
 for c in CASES:
  dest=INP/(c['run']+'.npy')
  if dest.exists():continue
  rd=daily.archive.hourly_lpi_archive_dir(root)/'2026'/c['run'];a=np.lib.format.open_memmap(dest,mode='w+',dtype='float32',shape=(24,len(KEYS),*SHAPE))
  for h in range(1,25):
   src=rd/f'f{h:03}.npz';meta=json.loads(src.with_suffix('.json').read_text());sha=p.digest(src);assert sha==meta['sha256']
   with np.load(src) as z:
    assert str(z['grid_hash'].item())==daily.ST.man['grid_hash'];assert str(z['formula_version'].item())==p.VERSION
    for k,key in enumerate(KEYS):a[h-1,k]=old.unpack_selected(z,key,IDS).reshape(SHAPE)
   sources.append(dict(path=str(src),sha256=sha))
  a.flush();del a
  if c['index']%10==0:log(f'Raw ingredient cache {c["index"]+1}/{len(CASES)}')
 if sources:p.write_csv(OUT/'ingredient_sources.csv',sources)

def components(a):
 li=old.descending_ramp(a[:,0],1.,-5.);cape=old.ramp(a[:,1],75.,800.);rh=old.ramp(a[:,2],45.,80.);dep=old.ramp(a[:,3],35.,150.)
 mid=old.ramp(a[:,4],35.,75.);u=old.ramp(a[:,5],.005,.05);rain=np.maximum(old.ramp(a[:,6],.05,1.5),old.ramp(a[:,7],.02,.8));charge=np.sqrt(rh*dep)
 surface=np.clip(a[:,8]/100,0,1);sub=np.clip(a[:,9]/100,0,1);finite=np.isfinite(a).all(1)
 values=dict(li=li,cape=cape,rh=rh,dep=dep,mid=mid,u=u,rain=rain,charge=charge,surface=surface,sub=sub)
 return {**{k:np.where(finite,v,np.nan) for k,v in values.items()},'finite':finite}

def hourly_score(x,c):
 family=c['family'];li=x['li'];cape=x['cape'];C=x['charge'];M=x['mid'];U=x['u'];R=x['rain'];a=c.get('a',.25);floor=c.get('floor',.2);uw=c.get('uweight',.55);power=c.get('power',1.)
 I=(1-a)*li+a*cape;T=uw*U+(1-uw)*R
 if family=='baseline':
  I=li*(.8+.2*cape);moist=C*(.85+.15*M);q=I*moist*np.sqrt(np.maximum(R,U*C));q[(I<.02)|(moist<.02)]=0
 elif family=='additive':q=.35*I+.25*C+.25*U+.15*R
 elif family=='soft_gate':q=(floor+(1-floor)*I)*(.4*C+.4*U+.2*R)*(floor+(1-floor)*(.5*C+.5*M))
 elif family=='geometric':q=((.45*I**power+.30*C**power+.25*T**power))**(1/power)*(floor+(1-floor)*I)
 elif family=='bottleneck':q=(1-floor)*np.minimum(np.minimum(I,C),np.maximum(U,R))+floor*(I+C+T)/3
 elif family=='noisy_or':
  route1=I*C*U;route2=li*(floor+(1-floor)*C)*R;route3=cape*C*np.maximum(U,R)*.6
  q=1-(1-route1)*(1-route2)*(1-route3)
 elif family=='ascent_first':q=(uw*U+(1-uw)*R*I)*(floor+(1-floor)*I)*(floor+(1-floor)*C)
 elif family=='instability_first':q=I*(floor+(1-floor)*np.maximum(U,R))*(floor+(1-floor)*(.6*C+.4*M))
 elif family=='dry_elevated':q=I*C*(uw*U+(1-uw)*R)*(1+power*(1-x['sub'])*U)/(1+power)
 elif family=='stratiform_veto':q=I*(floor+(1-floor)*C)*np.maximum(U,R*I**power)
 elif family=='lagged_ingredients':
  past=li.copy()
  for shift in range(1,int(c.get('lag',3))+1):past=np.fmax(past,np.concatenate((li[:1].repeat(shift,axis=0),li[:-shift]),axis=0))
  q=((1-a)*li+a*past)*(floor+(1-floor)*C)*np.maximum(U,R)
 elif family in ('daily_envelope','storm_dose','persistent_peak'):q=li*(.8+.2*cape)*C*(.85+.15*M)*np.sqrt(np.maximum(R,U*C))
 else:raise ValueError(family)
 q=np.clip(q,0,1);q[~x['finite']]=np.nan;return q

def aggregate(q,x,c,sl):
 family=c['family'];v=q[sl]
 if family=='daily_envelope':
  a=c.get('a',.25);floor=c.get('floor',.2);I=((1-a)*x['li'][sl]+a*x['cape'][sl]);I=np.nanmax(I,axis=0);C=np.nanmax(x['charge'][sl],axis=0);U=np.nanmax(x['u'][sl],axis=0);R=np.nanmax(x['rain'][sl],axis=0)
  return I*(floor+(1-floor)*C)*np.maximum(U,R)
 if family=='storm_dose':
  count=np.isfinite(v).sum(0);dose=-np.expm1(np.nansum(np.log1p(-np.clip(v,0,.999)),axis=0)/np.maximum(1.,c.get('dose',.4)*count));return np.where(count>0,dose,np.nan)
 if family=='persistent_peak':return (1-c.get('dose',.4))*np.nanmax(v,axis=0)+c.get('dose',.4)*np.nanmean(np.sqrt(v),axis=0)**2
 return np.nanmax(v,axis=0)

def engineering(x,sl):
 z={k:v[sl] for k,v in x.items() if k!='finite'};li=z['li'];C=z['charge'];U=z['u'];R=z['rain'];cape=z['cape']
 maximum=lambda v:np.nanmax(v,axis=0)
 mean=lambda v:np.nanmean(v,axis=0)
 count=x['finite'][sl].sum(0)
 fraction=lambda v:np.divide(v.sum(0),count,out=np.full(SHAPE,np.nan),where=count>0)
 return np.stack([maximum(li),maximum(cape),maximum(z['rh']),maximum(z['dep']),maximum(z['mid']),maximum(U),maximum(R),mean(li),mean(C),mean(U),fraction(li>.25),fraction(U>.2),maximum(li*C),maximum(li*C*U),maximum(li*C*R),maximum(cape*C*U),maximum(li*C*U*(1-z['sub'])),maximum(z['surface']),maximum(z['sub']),maximum(li*U)]).astype('float32')



def create_manual(configs,tag,features=False):
 cp=OUT/(tag+'_hist.npz')
 if cp.exists():return np.load(cp)
 nc=len(configs);dh=np.zeros((len(CASES),2,nc,2,200));bh=dh.copy();masks=[];block_masks=[];hour_counts=[];sh=np.zeros_like(dh);sb=np.zeros_like(dh)
 if features:
  fx=np.lib.format.open_memmap(OUT/'daily_features.npy',mode='w+',dtype='float32',shape=(len(CASES),20,N))
  bx=np.lib.format.open_memmap(OUT/'block_features.npy',mode='w+',dtype='float32',shape=(len(CASES),8,20,N))
 for c in CASES:
  i=c['index'];a=np.load(INP/(c['run']+'.npy'),mmap_mode='r');x=components(a);count=x['finite'].sum(0);complete=count>=1;hour_counts.append(count.ravel()[IX])
  # One input-based mask for every candidate, not a candidate-dependent sample.
  support=p.smooth(np.where(complete,1.,np.nan),20).ravel()[IX];obs=daily.ST.targets[c['obs'],2];target=np.max(obs,axis=0);mask=np.isfinite(support)&(obs!=255).all(0);masks.append(mask);bmask=[]
  for j in range(8):
   present=x['finite'][j*3:j*3+3].any(0);support_b=p.smooth(np.where(present,1.,np.nan),20).ravel()[IX];bmask.append(mask&np.isfinite(support_b))
  block_masks.append(bmask)
  if features:
   ef=engineering(x,slice(None));ef[:,~complete]=np.nan
   for k in range(20):fx[i,k]=p.smooth(ef[k],20).ravel()[IX]
   for j in range(8):
    bf=engineering(x,slice(j*3,j*3+3));bf[:,~x['finite'][j*3:j*3+3].any(0)]=np.nan
    for k in range(20):bx[i,j,k]=p.smooth(bf[k],20).ravel()[IX]
  for k,cfg in enumerate(configs):
   q=hourly_score(x,cfg);ds=aggregate(q,x,cfg,slice(None));ds[~complete]=np.nan;ds=p.smooth(ds*100,cfg['scale']).ravel()[IX]
   bs=[]
   for j in range(8):
    v=aggregate(q,x,cfg,slice(j*3,j*3+3));v[~complete]=np.nan;bs.append(p.smooth(v*100,cfg['scale']).ravel()[IX])
   for d,dm in enumerate(daily.DOM.values()):
    dh[i,d,k]=daily.hist(ds,target,mask&dm);sh[i,d,k]=daily.hist(ds,target,mask&dm&(count.ravel()[IX]>=12))
    for j in range(8):
     bh[i,d,k]+=daily.hist(bs[j],obs[j],bmask[j]&dm)/8
     sb[i,d,k]+=daily.hist(bs[j],obs[j],bmask[j]&dm&(count.ravel()[IX]>=12))/8
  del x,a
  if i%10==0:log(f'{tag} recipes {i+1}/{len(CASES)}')
 if features:fx.flush();bx.flush()
 np.savez_compressed(cp,dh=dh,bh=bh,mask=np.array(masks),block_mask=np.array(block_masks),hour_counts=np.array(hour_counts),sensitivity_dh=sh,sensitivity_bh=sb);return np.load(cp)


def fold_ids(fold,stage):return [c['index'] for c in CASES if daily.run_ok(c,fold) and daily.split(c['day'],fold)==stage]
def summarize(hist,fold,d,stage,k,prob=False):
 cal=(np.arange(1000)+.5)/1000 if prob else old.pava_probabilities(old.Histogram(*hist[fold_ids(fold,'train'),d,k].sum(0)))
 return p.histmetrics(hist[fold_ids(fold,stage),d,k].sum(0),cal)

def screen_manual(data,configs):
 rows=[]
 for fold in daily.FOLDS:
  for d,dom in enumerate(daily.DOM):
   for k,cfg in enumerate(configs):
    row=dict(fold=fold,domain=dom,name=cfg['name'],family=cfg['family'],scale=cfg['scale'])
    for dur,key in [('24h','dh'),('3h','bh')]:
     val=summarize(data[key],fold,d,'validate',k);ref=summarize(data[key],fold,d,'validate',0)
     row['brier_'+dur]=val['brier'];row['auc_'+dur]=val['auc'];row['gain_'+dur]=1-val['brier']/ref['brier'];row['auc_delta_'+dur]=val['auc']-ref['auc']
    rows.append(row)
 return rows

def fit_one(X,Y,ridge,signed):
 # Fixed physical scales; ridge penalizes ingredient weights, not the intercept.
 X=np.column_stack((np.ones(len(X)),X)).astype('float64');Y=Y.astype('float64');n=len(Y)
 def fun(beta):
  z=X@beta;res=expit(z)-Y
  return (np.sum(np.logaddexp(0,z)-Y*z)/n+ridge*np.sum(beta[1:]**2),X.T@res/n+np.r_[0,2*ridge*beta[1:]])
 start=np.zeros(X.shape[1]);rate=np.clip(Y.mean(),1e-4,1-1e-4);start[0]=np.log(rate/(1-rate))
 bounds=[(None,None)]+([(None,None)] if signed else [(0,None)])*(X.shape[1]-1)
 fit=minimize(fun,start,jac=True,method='L-BFGS-B',bounds=bounds,options={'maxiter':250,'ftol':1e-9});assert fit.success,fit.message
 return fit.x


def learned(manual,configs,tag):
 cp=OUT/(tag+'_models.npz');mp=OUT/(tag+'_model_coefficients.json')
 if cp.exists():return np.load(cp),json.loads(mp.read_text())
 fx=np.load(OUT/'daily_features.npy',mmap_mode='r');bx=np.load(OUT/'block_features.npy',mmap_mode='r');mask=manual['mask']
 # Store per-fold predictions as histograms; each model sees only its own earlier training data.
 dh=np.zeros((3,len(CASES),2,len(configs),2,1000));bh=dh.copy();sh=dh.copy();sb=dh.copy();coefs=[]
 rr,cc=np.unravel_index(IX,SHAPE);sample=(rr%2==0)&(cc%2==0)
 for fi,fold in enumerate(daily.FOLDS):
  train=fold_ids(fold,'train')
  for k,cfg in enumerate(configs):
   cols=MODEL[cfg['family']];ridge=cfg['ridge'];signed=cfg['family']=='signed_additive';params={}
   for d,domain in enumerate(daily.DOM):
    for duration,fields in [('24h',fx),('3h',bx)]:
     xx=[];yy=[]
     for i in train:
      use=mask[i]&sample&daily.DOM[domain]
      if duration=='24h':
       xx.append(fields[i,cols][:,use].T);yy.append(daily.ST.targets[CASES[i]['obs'],2].max(0)[use])
      else:
       for j in range(8):
        use=manual['block_mask'][i,j]&sample&daily.DOM[domain]
        xx.append(fields[i,j,cols][:,use].T);yy.append(daily.ST.targets[CASES[i]['obs'][j],2][use])
     X=np.concatenate(xx);Y=np.concatenate(yy);beta=fit_one(X,Y,ridge,signed);params[d,duration]=beta
     coefs.append(dict(fold=fold,domain=domain,name=cfg['name'],duration=duration,features=[FEATURES[j] for j in cols],intercept=float(beta[0]),weights=beta[1:].tolist(),training_rows=len(Y),training_days=len(train),ridge=ridge,converged=True))
   for c in CASES:
    i=c['index'];use=mask[i];obs=daily.ST.targets[c['obs'],2];target=obs.max(0)
    for d,dm in enumerate(daily.DOM.values()):
     probs=expit(params[d,'24h'][0]+fx[i,cols].T@params[d,'24h'][1:])
     bp=[expit(params[d,'3h'][0]+bx[i,j,cols].T@params[d,'3h'][1:]) for j in range(8)]
     dh[fi,i,d,k]=daily.hist(probs,target,use&dm,1000,True);sh[fi,i,d,k]=daily.hist(probs,target,use&dm&(manual['hour_counts'][i]>=12),1000,True)
     for j in range(8):
      block_use=manual['block_mask'][i,j]&dm
      bh[fi,i,d,k]+=daily.hist(bp[j],obs[j],block_use,1000,True)/8
      sb[fi,i,d,k]+=daily.hist(bp[j],obs[j],block_use&(manual['hour_counts'][i]>=12),1000,True)/8
   log(f'{tag}: {fold} {cfg["name"]}')
 np.savez_compressed(cp,dh=dh,bh=bh,sensitivity_dh=sh,sensitivity_bh=sb);p.write_json(mp,coefs);return np.load(cp),coefs

def screen_models(data,configs,reference):
 rows=[]
 for fi,fold in enumerate(daily.FOLDS):
  for d,dom in enumerate(daily.DOM):
   for k,cfg in enumerate(configs):
    row=dict(fold=fold,domain=dom,name=cfg['name'],family=cfg['family'],scale=20)
    for dur,key in [('24h','dh'),('3h','bh')]:
     val=summarize(data[key][fi],fold,d,'validate',k,True);ref=summarize(reference[key],fold,d,'validate',0)
     row['brier_'+dur]=val['brier'];row['auc_'+dur]=val['auc'];row['gain_'+dur]=1-val['brier']/ref['brier'];row['auc_delta_'+dur]=val['auc']-ref['auc']
    rows.append(row)
 return rows

def promising(rows):
 names=sorted({r['name'] for r in rows});candidates=[]
 for name in names:
  rr=[r for r in rows if r['name']==name and r['domain']=='all_bc' and r['scale']==20]
  if len(rr)!=3:continue
  gain=np.mean([r['gain_24h'] for r in rr]);consistent=sum(r['gain_24h']>0 for r in rr)
  if gain>.01 and consistent>=2 and min(r['gain_24h'] for r in rr)>-.05 and all(r['auc_delta_24h']>=-.02 for r in rr):candidates.append(dict(name=name,family=rr[0]['family'],gain=float(gain)))
 candidates.sort(key=lambda r:-r['gain']);best=[]
 for row in candidates:
  if row['family'] not in [r['family'] for r in best]:best.append(row)
  if len(best)==2:break
 return best

def refinements(winners):
 rng=np.random.default_rng(SEED);configs=[]
 for winner in winners:
  family=winner['family']
  if family in MODEL:continue
  for j in range(30):
   configs.append(dict(name=f'{family}_refine{j:02}',family=family,scale=20,a=float(rng.uniform(.02,.65)),floor=float(rng.uniform(0,.5)),uweight=float(rng.uniform(.2,.95)),power=float(rng.uniform(.25,2.5)),dose=float(rng.uniform(.1,.8)),lag=int(rng.integers(1,7))))
 return configs

def choose(rows):
 # Daily-first selection across validation periods; test results do not enter this function.
 summary=[]
 for name in sorted({r['name'] for r in rows}):
  rr=[r for r in rows if r['name']==name and r['domain']=='all_bc' and r['scale']==20]
  if len(rr)!=3 or any(r['auc_delta_24h']<-.02 for r in rr):continue
  summary.append(dict(name=name,family=rr[0]['family'],ratio24=float(np.mean([1-r['gain_24h'] for r in rr])),ratio3=float(np.mean([1-r['gain_3h'] for r in rr]))))
 best=min(summary,key=lambda r:r['ratio24']);close=[r for r in summary if r['ratio24']<=best['ratio24']*1.01];tie=min(close,key=lambda r:r['ratio3']);chosen=tie if tie['ratio3']<=best['ratio3']*.98 else best
 return dict(chosen=chosen,best_daily=best,close_candidates=close,all_validation_summaries=sorted(summary,key=lambda r:r['ratio24']))


def final_tests(manual,configs,models,modelconfigs,refdata=None,refconfigs=None,moremodels=None,moremodelconfigs=None):
 # Only called after validation selection has been written and locked.
 groups=[(manual,configs,False),(models,modelconfigs,True)]
 if refdata is not None:groups.append((refdata,refconfigs,False))
 if moremodels is not None:groups.append((moremodels,moremodelconfigs,True))
 rows=[];diffs=[]
 for data,defs,isprob in groups:
  for fi,fold in enumerate(daily.FOLDS):
   ids=fold_ids(fold,'test');dtrain=fold_ids(fold,'train')
   for d,domain in enumerate(daily.DOM):
    for k,cfg in enumerate(defs):
     for dur,key in [('24h','dh'),('3h','bh')]:
      hh=data[key][fi] if isprob else data[key];h=hh[ids,d,k]
      cal=(np.arange(1000)+.5)/1000 if isprob else old.pava_probabilities(old.Histogram(*hh[dtrain,d,k].sum(0)))
      m=p.histmetrics(h.sum(0),cal);rows.append(dict(fold=fold,domain=domain,name=cfg['name'],family=cfg['family'],scale=cfg.get('scale',20),duration=dur,stage='test',days=len(ids),**m))
      refh=manual[key][ids,d,0];refcal=old.pava_probabilities(old.Histogram(*manual[key][dtrain,d,0].sum(0)))
      loss=lambda hh,pc:(hh[:,1]*(1-pc)**2+(hh[:,0]-hh[:,1])*pc*pc).sum(1)
      dl=loss(h,cal)-loss(refh,refcal);nn=h[:,0].sum(1);assert np.allclose(nn,refh[:,0].sum(1));rng=np.random.default_rng(SEED);q={};nonempty=nn>0;dl=dl[nonempty];nn=nn[nonempty]
      for bl in (1,3):
       n=len(nn);bi=((rng.integers(0,n,(2000,int(np.ceil(n/bl))))[:,:,None]+np.arange(bl))%n).reshape(2000,-1)[:,:n];q[bl]=np.quantile(dl[bi].sum(1)/nn[bi].sum(1),[.025,.975])
      ref=p.histmetrics(refh.sum(0),refcal)
      diffs.append(dict(fold=fold,domain=domain,name=cfg['name'],duration=dur,delta_brier=m['brier']-ref['brier'],gain_percent=100*(1-m['brier']/ref['brier']),delta_auc=m['auc']-ref['auc'],ci_day_low=q[1][0],ci_day_high=q[1][1],ci_3day_low=q[3][0],ci_3day_high=q[3][1]))
 return rows,diffs

def main():
 warnings.filterwarnings('ignore',message='All-NaN slice encountered');warnings.filterwarnings('ignore',message='Mean of empty slice')
 protocol=dict(freeze=daily.ST.man['cutoff'],target='24h occurrence within30km; day1; allBC',cohort='Same62 full-file12Z ingredient runs; any available hourly data with all10 fields finite at that hour; daily95% spatial kernel support; each3hblock evaluated on its own common available-hour mask; sensitivity restricts centres toatleast12validhours',reference='Reconstructed current hourly recipe; daily max; sigma20km; training-only PAVA',folds=daily.FOLDS,manual_families=13,learned_families=list(MODEL),stage1_ridges=[.0005,.003,.02],refinement='Top2 validation-promising distinct families;30 randomized manual parameters or four additional local learned regularization values per family; fixed seed',promising_rule='Mean validation daily gain>1%,positivein2of3,minfoldgain>-5%,eachAUCdelta>=-.02',selection='Average relative validation daily Brier across3folds;1%daily tie tolerance;2%3hgain to change choice',holdout='Test scores reported only after candidate lock; historical data already inspected, so retrospective',source_predictors_only=True,no_forecaster_labels=True,no_day2_hourly_ingredients=True)
 if not (OUT/'protocol.json').exists():jwrite('protocol.json',protocol)
 prepare_inputs();configs=base_configs();jwrite('stage1_configs.json',configs);manual=create_manual(configs,'stage1',True)
 modelconfigs=[dict(name=f'{family}_ridge{ridge:g}',family=family,ridge=ridge,scale=20) for family in MODEL for ridge in (.0005,.003,.02)]
 jwrite('stage1_model_configs.json',modelconfigs);models,coefs=learned(manual,modelconfigs,'stage1')
 screen=screen_manual(manual,configs)+screen_models(models,modelconfigs,manual);p.write_csv(OUT/'stage1_validation.csv',screen);winners=promising(screen);jwrite('promising_families.json',winners);log(f'VALIDATION PROMISING {winners}')
 refs=refinements(winners);refdata=None;moremodels=None;morecfg=[]
 if refs:
  jwrite('refinement_configs.json',refs);refdata=create_manual(refs,'refinement');
  # Refinement reference is stage1 baseline, not the first refined recipe.
  rr=screen_manual(refdata,refs)
  for row in rr:
   d=list(daily.DOM).index(row['domain'])
   for dur,key in [('24h','dh'),('3h','bh')]:
    reference=summarize(manual[key],row['fold'],d,'validate',0);row['gain_'+dur]=1-row['brier_'+dur]/reference['brier'];row['auc_delta_'+dur]=row['auc_'+dur]-reference['auc']
  p.write_csv(OUT/'refinement_validation.csv',rr);screen+=rr
 for win in winners:
  if win['family'] in MODEL:
   morecfg += [dict(name=f'{win["family"]}_refine_ridge{r:g}',family=win['family'],ridge=r,scale=20) for r in (.0001,.0002,.001,.002)]
 if morecfg:
  jwrite('refinement_model_configs.json',morecfg);moremodels,extra=learned(manual,morecfg,'refinement');rr=screen_models(moremodels,morecfg,manual);p.write_csv(OUT/'refinement_model_validation.csv',rr);screen+=rr
 lock=choose(screen);jwrite('selection_lock.json',lock);log(f'LOCKED SELECTION {lock["chosen"]}')
 rows,diffs=final_tests(manual,configs,models,modelconfigs,refdata,refs,moremodels,morecfg);p.write_csv(OUT/'test_metrics.csv',rows);p.write_csv(OUT/'test_differences.csv',diffs)
 inventory=[]
 for fold in daily.FOLDS:
  for stage in ('train','validate','test'):
   ids=fold_ids(fold,stage)
   for d,dom in enumerate(daily.DOM):
    h=manual['dh'][ids,d,0];inventory.append(dict(fold=fold,stage=stage,domain=dom,days=len(ids),first=min(CASES[i]['day'] for i in ids),last=max(CASES[i]['day'] for i in ids),active_days=int(sum(x[1].sum()>0 for x in h)),eligible_cells=float(h[:,0].sum(1).mean()),coverage_percent=float(h[:,0].sum(1).mean()/daily.DOM[dom].sum()*100)))
 p.write_csv(OUT/'inventory.csv',inventory);jwrite('cases.json',CASES);log('RADICAL REVIEW COMPLETE')
if __name__=='__main__':main()
