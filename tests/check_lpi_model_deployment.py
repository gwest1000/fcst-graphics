"""Deployment parity, missing-data behavior, direct daily verification and metadata."""
import sys,json,pickle,tempfile,datetime as dt
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import lpi_model as l
import run_lpi_experiments_20261007 as e
import make_hrdps_west_lightning as lightning
import automate_lpi_verification as v
checks={}
# Full-grid reconstruction must match precomputed research features, not just a self-written example.
for i in (0,30,61):
 raw=np.load(e.r.INP/(e.r.CASES[i]['run']+'.npy'))
 for duration,sl in [('24h',slice(None)),('3h',slice(0,3))]:
  f,count=l.features(raw[sl]);reference=e.FX[i] if duration=='24h' else e.BX[i,0]
  actual=f.reshape(20,-1)[:,e.r.IX];assert np.array_equal(np.isfinite(actual),np.isfinite(reference));assert np.allclose(actual,reference,atol=1e-6,rtol=0,equal_nan=True)
checks['feature_engineering_matches_research_full_grid_three_season_cases']=True
m=pickle.loads((e.OUT/'random_network_0.001_summer_24h.pkl').read_bytes());model=dict(models={'24h':{k:m[k] for k in ('mean','sd','W','bias','beta')}});f=np.array(e.FX[0]);p=l.probability(f,'24h',model).ravel();good=np.isfinite(f).all(0);ref=100*e.predict(f[:,good].T,m);assert np.max(abs(p[good]-ref))<1e-4
checks['portable_network_matches_verified_candidate_probabilities']=True
blank=np.full((3,10,40,40),np.nan,np.float32);prob,count=l.infer(blank,'3h');assert np.isnan(prob).all() and not count.any()
assert not l.supported('west','12',3) and not l.supported('continental','00',3) and not l.supported('continental','12',27) and l.supported('continental','12',24)
checks['missing_inputs_stay_missing_and_activation_scope_is_enforced']=True
run=v.hrdps.RunInfo(cycle='12',stamp='20261007T12Z',init_time=dt.datetime(2026,10,7,12,tzinfo=dt.timezone.utc));window=v.first_full_12z_window(run)
with tempfile.TemporaryDirectory() as temp:
 root=Path(temp);paths={}
 lightning.set_model('continental')
 for h in window.included_hours:
  pth=lightning.save_lpi_cache(root/f'forecast_f{h:03d}.png',run,h,np.zeros((4,4)),np.zeros((4,4)),np.full((4,4),10.),1);paths[h]=pth
 path=Path(str(paths[24]).replace('_lpi.npz','_lpi24h.npz'));l.write_cache(path,run,'24h',np.zeros((4,4)),np.zeros((4,4)),np.full((4,4),77.),np.full((4,4),24))
 forecast=v.aggregate_daily_lpi(paths,window);assert np.all(forecast.potential==77) and forecast.formula_version.endswith('_24h')
 path.unlink();assert v.aggregate_daily_lpi(paths,window) is None
checks['daily_verification_uses_direct_probability_and_never_maxes_blocks']=True
Path('output/lpi_v4_deployment_checks.json').write_text(json.dumps({'passed':True,'checks':checks},indent=2));print(json.dumps(checks,indent=2))
