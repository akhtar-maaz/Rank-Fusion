import copy

import lightgbm as lgb
import numpy as np
import pytest

from harness.cv import grouped_kfold
from scripts.cv_augmented_lambdamart import summarize
from scripts.lambdamart_utils import default_params, fit
from scripts.robustness import SCENARIOS, perturb_features
from scripts.select_thirteenth_round import CONTROL, select, specifications, verify_all_scenarios
from submission.features import tree_features


def row(name,clean=.5,diagnostics=.45):
    values=[diagnostics]*5 if np.isscalar(diagnostics) else diagnostics
    folds=[dict(fold=i,train_ndcg=0,**{'train_ndcg@10':.7},
                scenarios=dict(clean=clean,affine=v,feature_dropout=v,noise=v)) for i,v in enumerate(values)]
    members=[dict(params={},trees=4,feature_mode='query_all',weight=.5) for _ in range(2)]
    return summarize(name,members,folds)


def baseline():
    control=row(CONTROL)
    return control,dict(selected=copy.deepcopy(control))


def test_simplex_is_complete_unique_and_native_bounded():
    specs=specifications()
    assert len(specs)==85 and specs[0]['name']==CONTROL
    assert len({tuple(zip(s['sources'],s['weights'])) for s in specs})==85
    assert max(sum(source[2] for source in s['sources']) for s in specs)==5600
    assert all(sum(s['weights'])==1 and all(w>0 for w in s['weights']) for s in specs)
    assert any(all(v[0] in ('v8','v11','v12') for v in s['sources']) for s in specs)
    assert any(len(s['sources'])==9 for s in specs)


def test_diagnostic_objective_wins_over_higher_clean_joint_score():
    control,old=baseline()
    cleaner=row('cleaner',clean=.504,diagnostics=.4513)
    robust=row('robust',diagnostics=.452)
    assert cleaner['selection_score']>robust['selection_score']
    selected=select([control,cleaner,robust],old)
    assert selected['promoted'] and selected['selected']['name']=='robust'


@pytest.mark.parametrize('condition',['clean','scenario','joint'])
def test_stronger_robustness_does_not_override_accuracy_and_joint_guards(condition):
    control,old=baseline()
    candidate=row('candidate',clean=.49 if condition=='clean' else .4976 if condition=='joint' else .5,
                  diagnostics=.452 if condition=='joint' else .47)
    if condition=='scenario':
        for fold in candidate['folds']:
            fold['scenarios']['affine']=.44
        candidate=summarize(candidate['name'],candidate['members'],candidate['folds'])
    result=select([control,candidate],old)
    assert not result['promoted'] and result['selected']['name']==CONTROL


def test_two_winning_folds_block_highest_mean_without_rescuing_runner_up():
    control,old=baseline()
    proposal=row('two_wins',diagnostics=[.46,.46,.4499,.4499,.4499])
    runner_up=row('qualifying_runner_up',diagnostics=.4515)
    assert select([control,runner_up],old)['promoted']
    result=select([control,proposal,runner_up],old)
    assert result['proposed']['name']=='two_wins' and result['promotion_gates']['improved_folds']==2
    assert not result['promoted'] and result['selected']['name']==CONTROL


def test_missing_fold_or_changed_control_fails_closed():
    control,old=baseline()
    missing=copy.deepcopy(control);missing['folds'].pop()
    with pytest.raises(ValueError,match='five distinct'):
        select([missing],old)
    changed=copy.deepcopy(control);changed['members'][0]['weight']=.4
    with pytest.raises(ValueError,match='control specifications'):
        select([changed],old)


@pytest.mark.parametrize('single',[True,False])
def test_native_diagnostic_export_matches_all_folds_and_scenarios(tmp_path,single):
    rng=np.random.RandomState(71)
    X=rng.normal(size=(160,3)).astype(np.float32);q=np.repeat(np.arange(20),8)
    y=(X[:,0]>0).astype(int);X[0,1]=np.nan
    dirs={v:tmp_path/v for v in ('a','b')}
    for folder in dirs.values():folder.mkdir()
    sources=[('a',0,5)] if single else [('a',0,5),('b',0,7)]
    modes={'a':'query_normalized','b':'query_all'}
    predictions={s:np.empty((4,len(y))) for s in sources}
    for fold,(tr,va) in enumerate(grouped_kfold(q)):
        for version,candidate,count in sources:
            model=fit(X[tr],y[tr],q[tr],default_params(7,min_data_in_leaf=2),count,modes[version])
            model.save_model(str(dirs[version]/f'candidate_{candidate}_fold_{fold}.txt'))
            for index,scenario in enumerate(SCENARIOS):
                perturbed=perturb_features(X[va],q[va],scenario)
                predictions[(version,candidate,count)][index,va]=model.predict(tree_features(perturbed,q[va],modes[version]),num_threads=1)
    setting=dict(sources=sources,weights=[1.] if single else [.25,.75])
    context=dict(prepared=tree_features(X,q,'query_all'),qids=q,labels=y,predictions=predictions,directories=dirs)
    checked=verify_all_scenarios(setting,context)
    assert len(checked)==5
    assert all([s['scenario'] for s in f['all_scenarios']]==list(SCENARIOS) for f in checked)
    assert max(s['maximum_absolute_error'] for f in checked for s in f['all_scenarios'])<=5e-12
