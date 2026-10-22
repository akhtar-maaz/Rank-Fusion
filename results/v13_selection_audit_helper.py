"""Complete TRAIN-only audit before V13 fitting or final DEV evaluation."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'results'))
from v12_selection_audit_helper import compare_recomputed
from scripts.cv_augmented_lambdamart import summarize
from scripts.select_thirteenth_round import specifications,select
from scripts.robustness import SCENARIOS

PROTOCOL='43434e62d179749065bb19fc7bd7100378fe16be'
SOURCE='6c81f8d147fb657bf51e2a6124b5f464f667150b'
BASELINE='f28b61880a56cac51566890671405db013bb3c94'


def read(path):return json.loads((ROOT/path).read_text())
def digest(path):return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args])


def audit():
    m=read('runs/v13_jobs.json');jobs=read('runs/v13_job_metadata.json')
    assert not m.get('pending_submission') and jobs[m['selection']]['job_state']=='F' and jobs[m['selection']]['Exit_status']=='0'
    assert (m['protocol_commit'],m['workflow_commit'],m['baseline_commit'])==(PROTOCOL,SOURCE,BASELINE)
    assert m['source_ci']['conclusion']=='success' and m['source_ci']['tests_passed']==m['hpc_source_tests']['passed']==185
    assert (ROOT/'docs/EXPERIMENTS_V13.md').read_bytes()==git('show',PROTOCOL+':docs/EXPERIMENTS_V13.md')
    for path,expected in m['source_sha256'].items():
        assert digest(path)==expected and (ROOT/path).read_bytes()==git('show',SOURCE+':'+path),path
    for name,expected in m['baseline_hashes'].items():assert digest('submission/'+name)==expected,name
    assert digest('results/neural_history.json')==m['baseline_neural_history_sha256']
    for path,expected in m['configuration_sha256'].items():assert digest(path)==expected,path
    identity=read('runs/v13_identity_preflight.json')
    assert digest('runs/v13_identity_preflight.json')==m['identity_preflight_sha256']
    assert identity['completed_before_measurement'] and identity['input_sha256']==m['fit_input_sha256']
    assert identity['source_sha256']==m['source_sha256'] and identity['protected_sha256']==m['baseline_hashes']
    assert identity['settings']==85 and identity['distinct_source_checkpoints']==11 and len(identity['cache_sha256'])==77
    report=read('results/lambdamart_cv_v13.json');settings=specifications()
    assert (report['rows'],report['queries'])==(723412,6000)
    assert len(report['results'])==85 and [r['name'] for r in report['results']]==[s['name'] for s in settings]
    parts={}
    for source in sorted({tuple(s) for setting in settings for s in setting['sources']}):
        version,candidate,count=source;key=(version,candidate)
        path=f'runs/cv_{version}/part_{candidate}.json'
        assert digest(path)==identity['cache_sha256'][path],path
        part=read(path);parts[key]=part
        assert (part['rows'],part['queries'])==(723412,6000) and count in part['candidate']['trees']
        assert all(sorted(f['fold'] for f in row['folds'])==list(range(5)) for row in part['results'])
        if version in ('v8','v11','v12'):
            assert part['candidate']==read(f'configs/{version}_candidates.json')[candidate]
    normalized=copy.deepcopy(report['results']);maximum=0.
    for row,setting in zip(normalized,settings):
        assert row['sources']==[list(s) for s in setting['sources']]
        members=row.get('members',[dict(row,weight=1.)])
        assert len(members)==len(setting['sources'])
        for member,(version,candidate,count),weight in zip(members,setting['sources'],setting['weights']):
            spec=parts[(version,candidate)]['candidate']
            assert member['params']==spec['params'] and member['trees']==count and member['weight']==weight
            assert member['feature_mode']==spec['feature_mode'] and member.get('augmentation')==spec.get('augmentation')
        assert sorted(f['fold'] for f in row['folds'])==list(range(5))
        assert all(set(f['scenarios'])==set(SCENARIOS) for f in row['folds'])
        expected=summarize(row['name'],members,row['folds'])
        for key in ('train_mean','validation_mean','validation_std','validation_train_ratio','diagnostic_mean','selection_score','scenario_means'):
            maximum=max(maximum,compare_recomputed(row[key],expected[key],'aggregate.'+key))
            row[key]=expected[key]
    recomputed=select(normalized,read('results/lambdamart_cv_release_v12.json'))
    for key,value in recomputed.items():maximum=max(maximum,compare_recomputed(report[key],value,'selection.'+key))
    native=report['native_equivalence'];assert [f['fold'] for f in native]==list(range(5))
    for fold in native:
        checks=fold['all_scenarios'];assert [c['scenario'] for c in checks]==list(SCENARIOS)
        assert all(c['maximum_absolute_error']<=5e-12 and abs(c['native_ndcg_10']-c['explicit_ndcg_10'])<=1e-12 for c in checks)
    part_hashes={path:expected for path,expected in identity['cache_sha256'].items() if '/part_' in path}
    return dict(protocol_commit=PROTOCOL,workflow_commit=SOURCE,baseline_commit=BASELINE,
        selected_on='released TRAIN only',new_dev_inspected=False,settings=85,source_checkpoints=11,
        selected=report['selected'],promoted=report['promoted'],promotion_gates=report['promotion_gates'],
        native_checks=20,maximum_native_error=max(c['maximum_absolute_error'] for f in native for c in f['all_scenarios']),
        numeric_audit=dict(absolute_tolerance=1e-12,maximum_recomputation_error=maximum,
            decisions_and_identities_exact=True,promotion_gates_unchanged=True),
        selection_job={m['selection']:jobs[m['selection']]},source_sha256=m['source_sha256'],
        input_sha256=m['fit_input_sha256'],protected_sha256=m['baseline_hashes'],
        cache_identity_sha256=m['identity_preflight_sha256'],source_part_sha256=part_hashes,
        selection_sha256=digest('results/lambdamart_cv_v13.json'))


if __name__=='__main__':
    result=audit();(ROOT/'results/v13_selection_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,promoted=result['promoted'],settings=85,native_checks=20,gates=result['promotion_gates'])))
