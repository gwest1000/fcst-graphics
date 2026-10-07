"""Check temporal separation, common support, selection, and direct score accuracy."""
import sys,json,pickle,csv
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import run_lpi_experiments_20261007 as e
O=e.OUT
checks={}
for path in O.glob('*.pkl'):
 m=pickle.loads(path.read_bytes());fold=next(f for f in sorted(e.D.FOLDS,key=len,reverse=True) if path.name.endswith(f+'_'+('3h' if '_3h.' in path.name else '24h')+'.pkl'))
 assert m['training_ids']==e.r.fold_ids(fold,'train') and m['converged']
 assert not set(m['training_ids'])&set(e.r.fold_ids(fold,'validate')+e.r.fold_ids(fold,'test'))
checks['fitted_models_converged_and_training_ids_exclude_validation_and_test']=True
for path in O.glob('*_test.npz'):
 h=np.load(path)['h'];duration='3h' if '_3h_' in path.name else '24h'
 for fi,fold in enumerate(e.D.FOLDS):
  ids=e.r.fold_ids(fold,'test')
  for si in range(2):assert np.allclose(h[fi,ids,:,si,0].sum(-1),e.reference(duration,si)[fi,ids,:,0].sum(-1))
checks['every_test_model_and_reference_uses_identical_support']=True
rows=list(csv.DictReader((O/'validation.csv').open()))
for x in rows:
 for k in ('gain','auc_delta','brier','auc'):x[k]=float(x[k])
ranks=[x for x in e.ranking(rows) if x['auc_ok']];best=min(ranks+[dict(name='previous_winner',ratio=1.)],key=lambda x:x['ratio']);close=[x for x in ranks+[dict(name='previous_winner',ratio=1.)] if x['ratio']<=best['ratio']*1.01]
r3={'previous_winner':1.}
for x in close:
 if x['name']!='previous_winner':r3[x['name']]=np.mean([1-r['gain'] for r in rows if r['name']==x['name'] and r['duration']=='3h'])
tie=min(close,key=lambda x:r3[x['name']]);chosen=tie if r3[tie['name']]<=r3[best['name']]*.98 else best
lock=json.loads((O/'selection_lock.json').read_text());assert chosen['name']==lock['chosen']['name']
checks['daily_first_selection_reproduces_from_validation_only']=True
n=chosen['name'];fold='summer';i=e.r.fold_ids(fold,'test')[0];m=pickle.loads((O/f'{n}_{fold}_24h.pkl').read_bytes());use=e.M['mask'][i];x=e.FX[i,:,use] # advanced indexing gives points x features
assert x.shape[1]==20
p=e.predict(x,m);y=e.D.ST.targets[e.r.CASES[i]['obs'],2].max(0)[use];direct=np.mean((p-y)**2);h=np.load(O/f'{n}_24h_test.npz')['h'][0,i,0,0];binned=e.P.histmetrics(h,e.CAL)['brier'];assert abs(direct-binned)<.001
checks['direct_probability_brier_agrees_with_histogram_within_bin_resolution']=True
assert (O/'selection_lock.json').stat().st_mtime<min(f.stat().st_mtime for f in O.glob('*_test.npz'))
checks['selection_lock_written_before_new_test_histograms']=True
result=dict(passed=True,checks=checks,direct_brier=float(direct),histogram_brier=float(binned));e.P.write_json(O/'independent_checks.json',result);print(json.dumps(result,indent=2))
