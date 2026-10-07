#!/usr/bin/env python3
"""Refit the fixed, validation-chosen architecture using the frozen BC cohort."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
import lpi_model as runtime
import run_lpi_experiments_20261007 as e

def main():
 lock=json.loads((e.OUT/'selection_lock.json').read_text());assert lock['chosen']['name']=='random_network_0.001'
 model=dict(version=runtime.VERSION,features=list(runtime.FEATURES),grid_hash=e.D.ST.man['grid_hash'],grid_resolution_km=5,feature_sigma_km=20,target_radius_km=30,supported=dict(model='continental',cycle='12',forecast_hours=list(range(3,25,3)),daily_hours=list(range(1,25))),architecture=dict(hidden_units=32,hidden_weights='fixed random',seed=20261007,output_loss='log_loss',ridge=.001),selection='random_network_0.001; chosen using historical validation, then frozen before retrospective tests',training_policy='Final refit on all62 frozen cases; no independent performance claim for refitted coefficients',archive_cutoff=e.D.ST.man['cutoff'],training_days=[c['day'] for c in e.r.CASES],models={})
 for duration,fields in [('24h',e.FX),('3h',e.BX)]:
  xx=[];yy=[]
  for i,c in enumerate(e.r.CASES):
   obs=e.D.ST.targets[c['obs'],2]
   for b in (range(8) if duration=='3h' else [None]):
    use=(e.M['mask'][i] if b is None else e.M['block_mask'][i,b])&e.SAMPLE
    xx.append((fields[i] if b is None else fields[i,b])[:,use].T);yy.append((obs.max(0) if b is None else obs[b])[use])
  X=np.concatenate(xx).astype('float64');Y=np.concatenate(yy).astype('float64');rng=np.random.default_rng(20261007)
  mean=X.mean(0);sd=np.maximum(X.std(0),.08);W=rng.normal(0,1/np.sqrt(20),(20,32));bias=rng.normal(0,.5,32)
  A=np.column_stack([np.ones(len(X)),X,np.tanh(((X-mean)/sd)@W+bias)]);n=len(Y)
  def fun(beta):
   z=A@beta;return np.mean(np.logaddexp(0,z)-Y*z)+.001*np.sum(beta[1:]**2),A.T@(expit(z)-Y)/n+np.r_[0,.002*beta[1:]]
  start=np.zeros(53);start[0]=np.log(Y.mean()/(1-Y.mean()));fit=minimize(fun,start,jac=True,method='L-BFGS-B',options={'maxiter':600,'ftol':1e-9,'gtol':1e-6});assert fit.success,fit.message
  model['models'][duration]=dict(mean=mean.tolist(),sd=sd.tolist(),W=W.tolist(),bias=bias.tolist(),beta=fit.x.tolist(),training_rows=len(Y),converged=True)
  print(duration,len(Y),'converged',flush=True)
 runtime.MODEL_PATH.write_text(json.dumps(model,indent=2)+'\n');print(runtime.MODEL_PATH,flush=True)
if __name__=='__main__':main()
