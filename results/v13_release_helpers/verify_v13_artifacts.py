"""Post-choice independent identity check and installation; never fits or selects."""
import hashlib
import json
from pathlib import Path
import shutil


def read(path):return json.loads(Path(path).read_text())
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    p=Path(path);t=p.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2)+'\n');t.replace(p)


def verify():
    m=read('runs/v13_jobs.json');audit=read('results/v13_selection_audit.json')
    assert m.get('selection_commit') and not m.get('pending_submission')
    assert digest('results/v13_selection_audit.json')==m['selection_audit_sha256']
    assert digest('results/lambdamart_cv_v13.json')==audit['selection_sha256']
    for path,expected in m['source_sha256'].items():assert digest(path)==expected,path
    for path,expected in m['completion_helper_sha256'].items():assert digest(path)==expected,path
    for name,expected in m['baseline_hashes'].items():assert digest('submission/'+name)==expected,name
    assert digest('results/neural_history.json')==m['baseline_neural_history_sha256']
    assert not Path('results/dev_metrics_v13.json').exists(),'final DEV evaluation already exists; do not repeat'
    tree=read('results/lambdamart_cv_v13.json');promoted=tree['promoted']
    assert tree['selected']==audit['selected'] and promoted==audit['promoted']==m['tree_refit_required']
    if promoted:
        config=read('configs/v13_tree.json');assert config['selected']==tree['selected']
        first=read('runs/refit_v13/primary/fit.json');second=read('runs/refit_v13/independent/fit.json')
        mode=tree['selected']['feature_mode'];multiplier=1 if mode in ('raw','query_percentile') else 3 if mode=='query_all' else 2
        for r in (first,second):
            assert r['fit_splits']==['TRAIN','DEV'] and r['use_dev_for_fitting']
            assert r['query_ids_namespaced_by_split'] and r['dev_tree_metric_is_in_sample'] and not r['early_stopping_used']
            assert (r['rows'],r['queries'],r['raw_features'])==(958671,8000,136)
            assert r['actual_trees']==tree['selected']['trees'] and r['actual_features']==136*multiplier
            assert r['selected']==config['selected'] and r['selection_provenance']==config['selection_provenance']
            assert r['configuration_sha256']==digest('configs/v13_tree.json')==m['final_configuration_sha256']
            assert {entry['split']:entry['sha256'] for entry in r['inputs']}==m['fit_input_sha256']
            for path,expected in r['source_sha256'].items():assert digest(path)==expected==m['source_sha256'][path]
            assert r['python']=='3.9.21'
        for key in ('inputs','selected','configuration_sha256','source_sha256','versions','python'):
            assert first[key]==second[key],key
        h=digest('runs/refit_v13/primary/model.txt')
        assert h==digest('runs/refit_v13/independent/model.txt')==first['artifact_sha256']==second['artifact_sha256']
        # No submission file changes until both full fresh fits verify.
        shutil.copyfile('runs/refit_v13/primary/model.txt','submission/model.txt')
        first.update(shipped_artifact='submission/model.txt',shipped_sha256=h)
        write('results/tree_fit_v13.json',first);write('results/tree_fit_reproduced_v13.json',second)
        write('results/lambdamart_reproducibility.json',dict(artifact='submission/model.txt',original_sha256=h,
            reproduced_sha256=h,byte_identical=True,refitted=True,fit_splits=['TRAIN','DEV'],
            primary_fit_record='results/tree_fit_v13.json',independent_fit_record='results/tree_fit_reproduced_v13.json',
            configuration='configs/v13_tree.json',input_configuration_source_runtime_equal=True))
        provenance=dict(protocol='docs/EXPERIMENTS_V13.md',fit_splits=['TRAIN','DEV'],refitted_in_v13=True,
                        fit_record='results/tree_fit_v13.json',configuration='configs/v13_tree.json',cv_scope='V13 TRAIN only')
    else:
        h=m['baseline_hashes']['model.txt'];assert read('results/tree_fit_v10.json')['artifact_sha256']==h
        provenance=dict(protocol='docs/EXPERIMENTS_V13.md',fit_splits=['TRAIN','DEV'],refitted_in_v13=False,
            retained_from='V10',fit_record='results/tree_fit_v10.json',configuration='configs/v10_tree.json',
            cv_scope='V13 TRAIN-only comparison retained exact evaluated V12/V10 tree')
    for name in ('pointwise_model.pt','pairwise_model.pt','fusion_config.json','recipe.py','pointwise_pairwise.py'):
        assert digest('submission/'+name)==m['baseline_hashes'][name]
    assert digest('results/neural_history.json')==m['baseline_neural_history_sha256']
    tree['final_refit']=provenance;write('results/lambdamart_cv.json',tree)
    result=dict(tree_refitted=promoted,tree_sha256=h,selection_commit=m['selection_commit'],
        final_tree_provenance=provenance,neural_artifacts_history_fusion_core_recipe_preserved=True,
        no_v13_neural_fitting=True)
    write('results/verification_v13.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':verify()
