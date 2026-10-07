#!/usr/bin/env python3
import os,sys,json,pickle
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
import run_lpi_radical_review_20261006 as r
D=r.daily; P=r.p
OUT=Path('output/lpi_calibration/review_20261007');OUT.mkdir(exist_ok=True)
M=np.load(r.OUT/'stage1_hist.npz'); FX=np.load(r.OUT/'daily_features.npy',mmap_mode='r'); BX=np.load(r.OUT/'block_features.npy',mmap_mode='r')
REF=np.load(r.OUT/'boundary_models.npz'); RC=json.loads((r.OUT/'boundary_model_configs.json').read_text()); RK=next(i for i,c in enumerate(RC) if c['name']=='signed_additive_boundary_ridge5e-05')
rr,cc=np.unravel_index(r.IX,r.SHAPE); SAMPLE=(rr%2==0)&(cc%2==0)
PAIRS=[(0,3),(0,5),(0,6),(1,5),(2,5),(3,5),(4,6),(6,10),(5,18),(7,9),(8,9),(10,11),(12,6),(13,18),(17,18)]

def transform(X,m):
 f=m['family']
 if f=='hinge':return np.column_stack([X]+[np.maximum(X-t,0) for t in (.25,.5,.75)])
 if f=='interactions':return np.column_stack([X,X*X]+[X[:,a]*X[:,b] for a,b in PAIRS])
 if f=='rbf':
  Z=(X-m['mean'])/m['sd']; C=m['centres']; dist=np.maximum((Z*Z).sum(1)[:,None]+(C*C).sum(1)-2*Z@C.T,0)
  return np.column_stack([X,np.exp(-dist/(2*m['width']**2))])
 if f=='random_network':return np.column_stack([X,np.tanh(((X-m['mean'])/m['sd'])@m['W']+m['bias'])])
 if f=='gaussian':
  Z=(X-m['mean'])/m['sd']; scores=[]
  for mu,inv,ld in m['gauss']:
   delta=Z-mu;scores.append(-.5*np.einsum('ij,jk,ik->i',delta,inv,delta)-.5*ld)
  return np.column_stack([np.clip((scores[1]-scores[0])/20,-20,20)])
 return X

def train(cfg,fold,duration):
 path=OUT/f"{cfg['name']}_{fold}_{duration}.pkl"
 if path.exists():return pickle.loads(path.read_bytes())
 xx=[];yy=[]
 for i in r.fold_ids(fold,'train'):
  for b in (range(8) if duration=='3h' else [None]):
   use=(M['mask'][i] if b is None else M['block_mask'][i,b])&SAMPLE
   xx.append((FX[i] if b is None else BX[i,b])[:,use].T)
   obs=D.ST.targets[r.CASES[i]['obs'],2]; yy.append((obs.max(0) if b is None else obs[b])[use])
 X=np.concatenate(xx).astype('float64');Y=np.concatenate(yy).astype('float64');rng=np.random.default_rng(20261007)
 m=dict(cfg);m.update(mean=X.mean(0),sd=np.maximum(X.std(0),.08),training_rows=len(Y),training_ids=r.fold_ids(fold,'train'))
 if cfg['family']=='rbf':m['centres']=((X-m['mean'])/m['sd'])[rng.choice(len(X),32,replace=False)];m['width']=cfg.get('width',3.)
 if cfg['family']=='random_network':m['W']=rng.normal(0,1/np.sqrt(20),(20,32));m['bias']=rng.normal(0,.5,32)
 if cfg['family']=='gaussian':
  Z=(X-m['mean'])/m['sd'];m['gauss']=[]
  for cls in (0,1):
   z=Z[Y==cls];cov=np.cov(z,rowvar=False);a=cfg['param'];cov=(1-a)*cov+a*np.eye(20);m['gauss'].append((z.mean(0),np.linalg.inv(cov),np.linalg.slogdet(cov)[1]))
 A=np.column_stack([np.ones(len(X)),transform(X,m)]);n=len(Y);ridge=cfg['param'] if cfg['family']!='gaussian' else .0001
 def fun(beta):
  z=A@beta;pr=expit(z)
  if cfg['family']=='brier':loss=np.mean((pr-Y)**2);g=2*(pr-Y)*pr*(1-pr)
  else:loss=np.mean(np.logaddexp(0,z)-Y*z);g=pr-Y
  return loss+ridge*np.sum(beta[1:]**2),A.T@g/n+np.r_[0,2*ridge*beta[1:]]
 start=np.zeros(A.shape[1]);rate=np.clip(Y.mean(),1e-5,1-1e-5);start[0]=np.log(rate/(1-rate))
 fit=minimize(fun,start,jac=True,method='L-BFGS-B',options={'maxiter':600,'ftol':1e-9,'gtol':1e-6});m['beta']=fit.x;m['converged']=bool(fit.success);m['message']=str(fit.message)
 if not fit.success:raise RuntimeError((cfg,fold,duration,fit.message))
 path.write_bytes(pickle.dumps(m));r.log(f"FIT {cfg['name']} {fold} {duration}");return m

def predict(X,m):
 out=np.empty(len(X))
 for st in range(0,len(X),5000):
  A=transform(X[st:st+5000],m);out[st:st+5000]=expit(m['beta'][0]+A@m['beta'][1:])
 return out

def histograms(cfg,duration,stage):
 path=OUT/f"{cfg['name']}_{duration}_{stage}.npz"
 if path.exists():return np.load(path)['h']
 h=np.zeros((3,len(r.CASES),2,2,2,1000)) # fold,case,domain,coverage,count/event,bin
 for fi,fold in enumerate(D.FOLDS):
  m=train(cfg,fold,duration)
  for i in r.fold_ids(fold,stage):
   obs=D.ST.targets[r.CASES[i]['obs'],2]
   for b in (range(8) if duration=='3h' else [None]):
    x=FX[i] if b is None else BX[i,b];use=M['mask'][i] if b is None else M['block_mask'][i,b];y=obs.max(0) if b is None else obs[b]
    prob=predict(np.nan_to_num(x.T),m)
    assert np.isfinite(prob).all() and ((prob>=0)&(prob<=1)).all()
    for di,dm in enumerate(D.DOM.values()):
     for si in range(2):h[fi,i,di,si]+=D.hist(prob,y,use&dm&((M['hour_counts'][i]>=12) if si else True),1000,True)/(8 if duration=='3h' else 1)
 np.savez_compressed(path,h=h);return h
CAL=(np.arange(1000)+.5)/1000

def reference(duration,si=0):return REF[('sensitivity_' if si else '')+('dh' if duration=='24h' else 'bh')][:,:,:,RK]
def metric(h):return P.histmetrics(h.sum(0),CAL)
def validation(cfg,duration):
 h=histograms(cfg,duration,'validate');rows=[]
 for fi,fold in enumerate(D.FOLDS):
  ids=r.fold_ids(fold,'validate');v=metric(h[fi,ids,0,0]);ref=metric(reference(duration)[fi,ids,0]);rows.append(dict(name=cfg['name'],family=cfg['family'],fold=fold,duration=duration,brier=v['brier'],auc=v['auc'],gain=1-v['brier']/ref['brier'],auc_delta=v['auc']-ref['auc']))
 return rows

def ranking(rows):
 out=[]
 for name in sorted({x['name'] for x in rows}):
  rs=[x for x in rows if x['name']==name and x['duration']=='24h']
  if len(rs)==3:out.append(dict(name=name,family=rs[0]['family'],ratio=float(np.mean([1-x['gain'] for x in rs])),auc_ok=all(x['auc_delta']>=-.02 for x in rs),gains=[x['gain'] for x in rs]))
 return sorted(out,key=lambda x:x['ratio'])
def main():
 protocol={'reference':RC[RK],'primary':'all BC 24h occurrence within30km, day1, sigma20km','folds':D.FOLDS,'selection':'mean relative daily validation Brier; AUC loss <=.02 per fold; within1% daily allow switch for >=2% better3h; previous winner competes','refine':'best two distinct new families on daily validation; three local parameter values each','reused_archive':'Retrospective; same62 cases and common masks as previous study. No untouched confirmation.','corridor':'Same all-BC fitted models scored over corridor (not separately fitted).'}
 P.write_json(OUT/'protocol.json',protocol)
 configs=[dict(name=f'{f}_{v:g}',family=f,param=v) for f,vs in [('hinge',[.0001,.001]),('interactions',[.0001,.001]),('brier',[.00001,.0001]),('rbf',[.0001,.001]),('random_network',[.0001,.001]),('gaussian',[.1,.6])] for v in vs]
 rows=[]
 for c in configs:rows+=validation(c,'24h');P.write_csv(OUT/'validation.csv',rows)
 ranks=ranking(rows);families=[]
 for v in ranks:
  if v['auc_ok'] and v['family'] not in families:families.append(v['family'])
  if len(families)==2:break
 for f in families:
  best=next(x for x in ranks if x['family']==f);c=next(x for x in configs if x['name']==best['name']);values=([max(.01,c['param']/2),min(.95,c['param']+.2),min(.98,c['param']+.35)] if f=='gaussian' else [c['param']/4,c['param']/2,c['param']*2])
  for v in values:
   new=dict(name=f'{f}_refine_{v:g}',family=f,param=v);configs.append(new);rows+=validation(new,'24h');P.write_csv(OUT/'validation.csv',rows)
 ranks=ranking(rows);eligible=[x for x in ranks if x['auc_ok']];best=min([dict(name='previous_winner',ratio=1.)]+eligible,key=lambda x:x['ratio']);close=[x for x in eligible if x['ratio']<=best['ratio']*1.01]
 # Also evaluate 3h for one representative per family, making architecture comparisons readable.
 short={x['name'] for x in close}
 for f in {x['family'] for x in ranks}:short.add(next(x['name'] for x in ranks if x['family']==f))
 for c in configs:
  if c['name'] in short:rows+=validation(c,'3h');P.write_csv(OUT/'validation.csv',rows)
 r3={name:float(np.mean([1-x['gain'] for x in rows if x['name']==name and x['duration']=='3h'])) for name in short};r3['previous_winner']=1.
 choices=close+([dict(name='previous_winner',ratio=1.)] if 1<=best['ratio']*1.01 else [])
 tie=min(choices,key=lambda x:r3[x['name']]);chosen=tie if r3[tie['name']]<=r3[best['name']]*.98 else best
 lock=dict(chosen=chosen,best_daily=best,three_hour_ratios=r3,ranking=ranks,refined_families=families);P.write_json(OUT/'selection_lock.json',lock);P.write_json(OUT/'configs.json',configs);r.log(f'LOCKED {chosen}')
 results=[]
 # Locked family representatives plus chosen; no refinement after these tests.
 evaluate={next(x['name'] for x in ranks if x['family']==f) for f in families+[x['family'] for x in ranks]};evaluate.add(chosen['name']);evaluate.discard('previous_winner')
 for c in configs:
  if c['name'] not in evaluate:continue
  for duration in ['24h','3h']:
   h=histograms(c,duration,'test')
   for fi,fold in enumerate(D.FOLDS):
    ids=r.fold_ids(fold,'test')
    for di,dom in enumerate(D.DOM):
     for si in range(2):
      a=h[fi,ids,di,si];b=reference(duration,si)[fi,ids,di];nn=a[:,0].sum(1);assert np.allclose(nn,b[:,0].sum(1));loss=lambda z:(z[:,1]*(1-CAL)**2+(z[:,0]-z[:,1])*CAL**2).sum(1)
      dl=loss(a)-loss(b);use=nn>0;dl=dl[use];nn=nn[use];rng=np.random.default_rng(20261007);ci={}
      for block in (1,3):
       n=len(nn);ix=((rng.integers(0,n,(2000,int(np.ceil(n/block))))[:,:,None]+np.arange(block))%n).reshape(2000,-1)[:,:n];ci[str(block)]=np.quantile(dl[ix].sum(1)/nn[ix].sum(1),[.025,.975]).tolist()
      ma=metric(a);mb=metric(b);results.append(dict(name=c['name'],family=c['family'],fold=fold,domain=dom,coverage='12hours' if si else 'available',duration=duration,n=float(nn.sum()),brier=ma['brier'],reference_brier=mb['brier'],gain_percent=100*(1-ma['brier']/mb['brier']),auc_delta=ma['auc']-mb['auc'],ci_day_low=ci['1'][0],ci_day_high=ci['1'][1],ci_3day_low=ci['3'][0],ci_3day_high=ci['3'][1]))
   P.write_csv(OUT/'test_results.csv',results)
 P.write_json(OUT/'checks.json',{'complete':True,'common_support_assertions_passed':True,'all_fits_converged':True,'test_selection_lock_precedes_test_evaluation':True});r.log('COMPLETE')
if __name__=='__main__':main()
