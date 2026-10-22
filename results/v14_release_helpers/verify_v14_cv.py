"""Verify the fixed, already selected V14 nested scorer on TRAIN OOF only."""
import hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import lightgbm as lgb
import numpy as np
from harness.cv import grouped_kfold
from harness.metrics import mean_ndcg_from_scores
from scripts.robustness import SCENARIOS,perturb_features
from submission.data import load_ltr
from submission.features import features_for_model
from submission.tree_ensemble import combine

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
started=time.perf_counter();m=json.loads(Path('runs/v14_jobs.json').read_text())
for p,h in m['source_sha256'].items():assert digest(p)==h,p
for p,h in m['configuration_sha256'].items():assert digest(p)==h,p
old=json.loads(Path('results/identity_preflight_v13.json').read_text())
for p,h in old['cache_sha256'].items():assert digest(p)==h,p
assert digest('data/train/train.txt')==m['fit_input_sha256']['TRAIN']
r=json.loads(Path('configs/v14_tree.json').read_text())['selected']
X,y,q=load_ltr('data/train/train.txt')[0];assert (len(y),len(set(q)),X.shape[1])==(723412,6000,136)
pred={}
for source in r['sources']:
 v,c,n=source;folder=Path('runs/cv_'+v);part=json.loads((folder/f'part_{c}.json').read_text())
 with np.load(folder/f'oof_{c}.npz') as a:
  assert np.array_equal(a['labels'],y) and np.array_equal(a['qids'],q) and tuple(a['scenarios'])==SCENARIOS
  index=list(a['trees']).index(n);pred[tuple(source)]=a['scores'][index].copy()
 assert pred[tuple(source)].shape==(4,len(y)) and np.isfinite(pred[tuple(source)]).all()
checks=[]
for fold,(_,va) in enumerate(grouped_kfold(q,k=5,seed=0)):
 models=[lgb.Booster(model_file=f'runs/cv_{v}/candidate_{c}_fold_{fold}.txt') for v,c,n in r['sources']]
 counts=[n for v,c,n in r['sources']]
 oldweights=[x['weight']*2 for x in r['members'][:6]]
 base=combine(models[:6],oldweights,counts[:6]);extra=combine(models[6:],[.5,.5],counts[6:])
 native=combine([base,extra],[.5,.5]);assert native.num_trees()==4800 and native.num_feature()==408
 scenarios=[]
 for i,name in enumerate(SCENARIOS):
  raw=X[va] if i==0 else perturb_features(X[va],q[va],name)
  scores=native.predict(features_for_model(native,raw,q[va]),num_threads=4)
  expected=sum(member['weight']*pred[tuple(source)][i,va] for member,source in zip(r['members'],r['sources']))
  error=float(np.max(np.abs(scores-expected)))
  actual=mean_ndcg_from_scores(scores,y[va],q[va]);reference=mean_ndcg_from_scores(expected,y[va],q[va])
  assert error<=5e-12 and abs(actual-reference)<=1e-12 and abs(actual-r['folds'][fold]['scenarios'][name])<=1e-12
  scenarios.append(dict(scenario=name,maximum_absolute_error=error,native_ndcg_10=actual,explicit_ndcg_10=reference))
 checks.append(dict(fold=fold,scenarios=scenarios));print(json.dumps(checks[-1]),flush=True)
result=dict(selected=r,selected_before_new_fitting=True,selection_type='known TRAIN-CV candidate after user priority clarification',
 checks=checks,native_checks=20,maximum_native_error=max(c['maximum_absolute_error'] for f in checks for c in f['scenarios']),
 source_sha256=m['source_sha256'],input_sha256=m['fit_input_sha256']['TRAIN'],cache_sha256=old['cache_sha256'],
 configuration_sha256=digest('configs/v14_tree.json'),rows=len(y),queries=len(set(q)),seconds=time.perf_counter()-started)
t=Path('results/native_cv_v14.pending.json');t.write_text(json.dumps(result,indent=2)+'\n');t.replace('results/native_cv_v14.json')
