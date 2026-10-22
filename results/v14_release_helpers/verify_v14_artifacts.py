"""Verify two fresh member groups, then compile with the retained base."""
import hashlib,json,shutil,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import lightgbm as lgb
import numpy as np
from submission.features import features_for_model
from submission.tree_ensemble import combine

def read(p):return json.loads(Path(p).read_text())
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,r):
 t=Path(p).with_suffix('.tmp');t.write_text(json.dumps(r,indent=2)+'\n');t.replace(p)
m=read('runs/v14_jobs.json');assert not m.get('pending_submission') and m.get('verification')
assert not Path('results/dev_metrics_v14.json').exists(),'do not repeat DEV'
for group in ('source_sha256','configuration_sha256','helper_sha256','verification_helper_sha256'):
 for p,h in m[group].items():assert digest(p)==h,p
for name,h in m['baseline_hashes'].items():assert digest('submission/'+name)==h,name
assert digest('results/neural_history.json')==m['baseline_neural_history_sha256']
cv=read('results/native_cv_v14.json');config=read('configs/v14_tree.json');newconfig=read('configs/v14_new_members.json')
assert cv['selected']==config['selected'] and cv['configuration_sha256']==digest('configs/v14_tree.json')
assert cv['native_checks']==20 and cv['maximum_native_error']<=5e-12
assert cv['source_sha256']==m['source_sha256'] and cv['input_sha256']==m['fit_input_sha256']['TRAIN']
assert cv['cache_sha256']==read('results/identity_preflight_v13.json')['cache_sha256']
assert [f['fold'] for f in cv['checks']]==list(range(5))
for fold in cv['checks']:
 assert [c['scenario'] for c in fold['scenarios']]==['clean','affine','feature_dropout','noise']
 assert all(c['maximum_absolute_error']<=5e-12 and abs(c['native_ndcg_10']-c['explicit_ndcg_10'])<=1e-12 for c in fold['scenarios'])
records=[read('runs/refit_v14/'+branch+'/fit.json') for branch in ('primary','independent')]
for r in records:
 assert r['selected']==newconfig['selected'] and r['selection_provenance']==newconfig['selection_provenance']
 assert r['configuration_sha256']==digest('configs/v14_new_members.json')
 assert r['fit_splits']==['TRAIN','DEV'] and r['use_dev_for_fitting'] and r['query_ids_namespaced_by_split'] and not r['early_stopping_used'] and r['dev_tree_metric_is_in_sample']
 assert (r['rows'],r['queries'],r['raw_features'],r['actual_trees'],r['actual_features'])==(958671,8000,136,1000,408)
 assert {x['split']:x['sha256'] for x in r['inputs']}==m['fit_input_sha256']
 assert r['python']=='3.9.21'
 for p,h in r['source_sha256'].items():assert digest(p)==h==m['source_sha256'][p]
for key in ('inputs','selected','configuration_sha256','source_sha256','versions','python'):
 assert records[0][key]==records[1][key],key
extra_hash=digest('runs/refit_v14/primary/model.txt')
assert extra_hash==digest('runs/refit_v14/independent/model.txt')==records[0]['artifact_sha256']==records[1]['artifact_sha256']
base=lgb.Booster(model_file='submission/model.txt');assert (base.num_trees(),base.num_feature())==(3800,408)
base_hash=digest('submission/model.txt');assert base_hash==read('results/tree_fit_v10.json')['artifact_sha256']
rng=np.random.RandomState(76414);X=rng.normal(size=(256,136)).astype(np.float32);X[::17,3]=np.nan;q=np.repeat(np.arange(16),16)
compiled=[];maximum=0.
for branch in ('primary','independent'):
 extra=lgb.Booster(model_file='runs/refit_v14/'+branch+'/model.txt')
 final=combine([base,extra],[.5,.5]);assert (final.num_trees(),final.num_feature())==(4800,408)
 transformed=features_for_model(final,X,q)
 actual=final.predict(transformed,num_threads=4)
 expected=.5*base.predict(features_for_model(base,X,q),num_threads=4)+.5*extra.predict(features_for_model(extra,X,q),num_threads=4)
 error=float(np.max(np.abs(actual-expected)));assert np.isfinite(actual).all() and error<=5e-12
 maximum=max(maximum,error);path='runs/refit_v14/'+branch+'/final_model.txt';final.save_model(path);compiled.append(digest(path))
assert compiled[0]==compiled[1] and compiled[0]!=base_hash
# Install only after independent full-group and full-composition checks passed.
shutil.copyfile('results/lambdamart_reproducibility.json','results/lambdamart_reproducibility_pre_v14.json')
shutil.copyfile('runs/refit_v14/primary/final_model.txt','submission/model.txt')
for name,h in m['baseline_hashes'].items():
 if name!='model.txt':assert digest('submission/'+name)==h
assert digest('results/neural_history.json')==m['baseline_neural_history_sha256']
write('results/tree_new_group_fit_v14.json',records[0]);write('results/tree_new_group_fit_reproduced_v14.json',records[1])
composition=dict(partial_refit=True,all_members_freshly_refitted=False,retained_base=dict(artifact_sha256=base_hash,trees=3800,weight=.5,fit_record='results/tree_fit_v10.json'),
 fresh_group=dict(trees=1000,weight=.5,artifact_sha256=extra_hash,primary='results/tree_new_group_fit_v14.json',independent='results/tree_new_group_fit_reproduced_v14.json'),
 artifact='submission/model.txt',artifact_sha256=compiled[0],independent_artifact_sha256=compiled[1],byte_identical=True,
 input_configuration_source_runtime_equal=True,fit_splits=['TRAIN','DEV'],rows=958671,queries=8000,raw_features=136,trees=4800,internal_features=408,
 maximum_synthetic_composition_error=maximum,configuration='configs/v14_tree.json',new_group_configuration='configs/v14_new_members.json',choice_commit=m['choice_commit'])
write('results/tree_composition_v14.json',composition)
write('results/lambdamart_reproducibility.json',dict(original_sha256=compiled[0],reproduced_sha256=compiled[1],byte_identical=True,partial_refit=True,
 all_members_freshly_refitted=False,retained_base_sha256=base_hash,new_member_group_sha256=extra_hash,input_configuration_source_runtime_equal=True,fit_splits=['TRAIN','DEV'],record='results/tree_composition_v14.json'))
current=dict(selected=config['selected'],selection_protocol=config['selection_provenance'],original_v13_gate_passed=False,
 native_equivalence=cv['checks'],final_refit=composition,cv_scope='same released TRAIN folds reused; retrospective known-result C2-first choice after user clarification')
write('results/lambdamart_cv_v14.json',current);write('results/lambdamart_cv.json',current)
write('results/verification_v14.json',dict(passed=True,choice_commit=m['choice_commit'],tree_sha256=compiled[0],new_group_byte_identical=True,final_composition_byte_identical=True,
 partial_refit=True,all_members_freshly_refitted=False,no_neural_fitting=True,neural_history_fusion_core_recipe_preserved=True,native_checks=20,composition=composition))
print(json.dumps(dict(passed=True,tree_sha256=compiled[0],trees=4800,partial_refit=True)),flush=True)
