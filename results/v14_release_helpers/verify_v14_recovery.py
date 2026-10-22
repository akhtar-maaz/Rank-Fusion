"""Use both completed stochastic fits; verify fixed export, not training-byte identity."""
import hashlib,json,re,shutil,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import lightgbm as lgb
import numpy as np
from submission.features import features_for_model
from submission.tree_ensemble import combine
from scripts.robustness import perturb_features

def read(p):return json.loads(Path(p).read_text())
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,r):
 t=Path(p).with_suffix('.tmp');t.write_text(json.dumps(r,indent=2)+'\n');t.replace(p)
m=read('runs/v14_jobs.json');recovery=m['recovery_1']
assert not m.get('pending_submission') and recovery.get('verification') and recovery['choice_pushed']
assert not Path('results/dev_metrics_v14.json').exists(),'never repeat DEV'
for group in (m['source_sha256'],m['configuration_sha256'],m['helper_sha256'],m['verification_helper_sha256'],recovery['configuration_sha256'],recovery['helper_sha256']):
 for p,h in group.items():assert digest(p)==h,p
for name,h in m['baseline_hashes'].items():assert digest('submission/'+name)==h,name
assert digest('results/neural_history.json')==m['baseline_neural_history_sha256']
audit=read('results/v14_variation_audit.json');cv=read('results/native_cv_v14.json')
assert audit['dev_not_evaluated'] and all(audit['identity_keys_equal'].values())
assert cv['native_checks']==20 and cv['maximum_native_error']<=5e-12 and cv['cache_sha256']==read('results/identity_preflight_v13.json')['cache_sha256']
assert cv['configuration_sha256']==digest('configs/v14_tree.json') and cv['source_sha256']==m['source_sha256'] and cv['input_sha256']==m['fit_input_sha256']['TRAIN']
records=[read('runs/refit_v14/'+b+'/fit.json') for b in ('primary','independent')];newconfig=read('configs/v14_new_members.json')
for r in records:
 assert r['selected']==newconfig['selected'] and r['configuration_sha256']==digest('configs/v14_new_members.json')
 assert (r['rows'],r['queries'],r['raw_features'],r['actual_trees'],r['actual_features'])==(958671,8000,136,1000,408)
 assert r['fit_splits']==['TRAIN','DEV'] and r['use_dev_for_fitting'] and r['query_ids_namespaced_by_split'] and not r['early_stopping_used']
 assert {x['split']:x['sha256'] for x in r['inputs']}==m['fit_input_sha256']
 for p,h in r['source_sha256'].items():assert digest(p)==h==m['source_sha256'][p]
for key in ('inputs','selected','configuration_sha256','source_sha256','versions','python'):assert records[0][key]==records[1][key]
assert records[0]['python']=='3.9.21'
paths=['runs/refit_v14/'+b+'/model.txt' for b in ('primary','independent')];hashes=[digest(p) for p in paths]
assert hashes==audit['model_sha256']==[r['artifact_sha256'] for r in records] and len(set(hashes))==2
blocks=[re.findall(r'(?ms)^Tree=\d+\n(.*?)(?=^Tree=\d+\n|^end of trees)',Path(p).read_text()) for p in paths]
assert len(blocks[0])==len(blocks[1])==1000
assert sum(a!=b for a,b in zip(blocks[0][:600],blocks[1][:600]))==599 and blocks[0][600:]==blocks[1][600:]
base_hash=digest('submission/model.txt');assert base_hash==read('results/tree_fit_v10.json')['artifact_sha256']
models=[lgb.Booster(model_file=p) for p in ['submission/model.txt']+paths]
assert [(x.num_trees(),x.num_feature()) for x in models]==[(3800,408),(1000,408),(1000,408)]
rng=np.random.RandomState(76414);X=rng.normal(size=(256,136)).astype(np.float32);X[::17,3]=np.nan;q=np.repeat(np.arange(16),16)
exports=[];maxerror=0.
for i in range(2):
 final=combine(models,[.5,.25,.25]);assert (final.num_trees(),final.num_feature())==(5800,408)
 for scenario in ('clean','affine','feature_dropout','noise'):
  raw=X if scenario=='clean' else perturb_features(X,q,scenario)
  feats=features_for_model(final,raw,q);actual=final.predict(feats,num_threads=4)
  expected=sum(w*model.predict(features_for_model(model,raw,q),num_threads=4) for w,model in zip((.5,.25,.25),models))
  error=float(np.max(np.abs(actual-expected)));assert np.isfinite(actual).all() and error<=5e-12
  assert np.array_equal(actual,final.predict(feats,num_threads=4));maxerror=max(maxerror,error)
 folder=Path(f'runs/recovery_v14/export_{i}');folder.mkdir(parents=True,exist_ok=True);p=folder/'model.txt';final.save_model(str(p));exports.append(digest(p))
assert exports[0]==exports[1] and exports[0]!=base_hash
shutil.copyfile('results/lambdamart_reproducibility.json','results/lambdamart_reproducibility_pre_v14.json')
shutil.copyfile('runs/recovery_v14/export_0/model.txt','submission/model.txt')
for name,h in m['baseline_hashes'].items():
 if name!='model.txt':assert digest('submission/'+name)==h
assert digest('results/neural_history.json')==m['baseline_neural_history_sha256']
for branch,r in zip(('primary','independent'),records):write('results/tree_new_group_fit_'+branch+'_v14.json',r)
composition=dict(artifact='submission/model.txt',artifact_sha256=exports[0],trees=5800,internal_features=408,raw_features=136,
 partial_refit=True,all_members_freshly_refitted=False,tree_training_byte_reproducible=False,
 retained_base=dict(artifact_sha256=base_hash,trees=3800,weight=.5,fit_record='results/tree_fit_v10.json'),
 fresh_groups=[dict(branch=b,artifact_sha256=h,trees=1000,weight=.25,fit_record='results/tree_new_group_fit_'+b+'_v14.json') for b,h in zip(('primary','independent'),hashes)],
 effective_weights=dict(retained_base=.5,query_bagging_primary=.125,query_bagging_independent=.125,balanced=.25),balanced_400_trees_exported_twice=True,
 fit_splits=['TRAIN','DEV'],original_rows=958671,original_queries=8000,inputs_config_source_runtime_equal=True,
 repeated_export_byte_identical=True,exports_share_same_three_fixed_inputs=True,independent_training_reproduction_claimed=False,
 repeated_export_sha256=exports,maximum_synthetic_arithmetic_error=maxerror,inference_deterministic=True,
 original_choice_commit=m['choice_commit'],recovery_choice_commit=recovery['choice_commit'],configuration='configs/v14_recovery_tree.json')
write('results/tree_composition_v14.json',composition)
write('results/lambdamart_reproducibility.json',composition)
config=read('configs/v14_recovery_tree.json');current=dict(selected=config['selected'],selection_protocol=config['selection_provenance'],
 original_v13_performance_gate_passed=False,original_v14_training_identity_gate_passed=False,parent_family_cv='results/v14_choice.json',
 parent_family_native_checks=cv['checks'],final_composition=composition,new_independent_cv_performance_measured=False)
write('results/lambdamart_cv_v14.json',current);write('results/lambdamart_cv.json',current)
write('results/verification_v14_recovery.json',dict(passed=True,recovery_choice_commit=recovery['choice_commit'],tree_sha256=exports[0],trees=5800,
 tree_training_byte_reproducible=False,original_verification_failed=True,all_completed_new_fits_included_without_accuracy_selection=True,
 export_reproducible_from_fixed_inputs=True,inference_deterministic=True,partial_refit=True,no_neural_fitting=True,composition=composition))
print(json.dumps(dict(passed=True,tree_sha256=exports[0],trees=5800,training_byte_reproducible=False,repeated_export_byte_identical=True)),flush=True)
