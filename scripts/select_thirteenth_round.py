"""Frozen robustness-first comparisons of complementary TRAIN-fold scorers."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lightgbm as lgb
import numpy as np

from harness.cv import grouped_kfold
from harness.metrics import mean_ndcg_from_scores
from scripts.ensemble_comparison import measure, verify_native
from scripts.robustness import SCENARIOS, perturb_features
from scripts.select_fifth_round import select as validate_rows_and_control
from submission.features import features_for_model
from submission.tree_ensemble import combine

ROOT = Path(__file__).resolve().parents[1]
CONTROL = 'cached_v12_control'


def specifications():
    basis = json.loads((ROOT/'configs/v13_blend_basis.json').read_text())
    assert basis['simplex_denominator']==4 and basis['minimum_control_units']==0
    control = [tuple(s) for s in basis['control_sources']]
    options = [(None,None,(4,0,0,0))]
    for bagging in basis['bagging_sources']:
        for balanced in basis['balanced_sources']:
            for a in range(5):
                for b in range(5-a):
                    for c in range(5-a-b):
                        options.append((bagging,balanced,(a,b,c,4-a-b-c)))
    settings,seen = [],set()
    for bagging,balanced,units in options:
        a,b,c,d = units
        sources = control.copy() if a else []
        weights = [w*a/4 for w in basis['control_weights']] if a else []
        for source,count in ((bagging,b),(balanced,c),(basis['normalization_source'],d)):
            if count:
                sources.append(tuple(source));weights.append(count/4)
        identity = tuple(zip(sources,weights))
        if identity in seen:
            continue
        seen.add(identity)
        name = CONTROL if units==(4,0,0,0) else 'complement_%d'%len(settings)
        settings.append(dict(name=name,sources=sources,weights=weights,basis_units=list(units)))
    if len(settings)!=85:
        raise ValueError('frozen simplex must have exactly85 distinct settings')
    return settings


def diagnostic_score(fold):
    return sum(fold['scenarios'][name] for name in SCENARIOS[1:])/3


def select(rows,baseline):
    validated=validate_rows_and_control(rows,baseline,control_name=CONTROL,control_key='v12_control')
    control=validated['v12_control']
    eligible=[r for r in rows if r['validation_mean']>=control['validation_mean']-.0025 and
              all(r['scenario_means'][s]>=control['scenario_means'][s]-.0025 for s in SCENARIOS[1:])]
    proposal=max(eligible,key=lambda r:(r['diagnostic_mean'],-r['trees'])) if eligible else control
    folds={f['fold']:f for f in control['folds']}
    gains=[dict(fold=f['fold'],diagnostic_gain=diagnostic_score(f)-diagnostic_score(folds[f['fold']]))
           for f in sorted(proposal['folds'],key=lambda f:f['fold'])]
    delta=proposal['diagnostic_mean']-control['diagnostic_mean']
    joint=proposal['selection_score']-control['selection_score']
    wins=sum(f['diagnostic_gain']>0 for f in gains)
    promoted=bool(eligible) and delta>=.001 and joint>=.0005 and wins>=3
    return dict(results=rows,selected=proposal if promoted else control,promoted=promoted,
        proposed=proposal,v12_control=control,eligible_settings=[r['name'] for r in eligible],
        promotion_gates=dict(candidate=proposal['name'],diagnostic_gain=delta,historical_joint_gain=joint,
            improved_folds=wins,paired_diagnostic_gains=gains,diagnostic_passed=delta>=.001,
            joint_passed=joint>=.0005,paired_folds_passed=wins>=3,passed=promoted),
        selection_protocol=dict(split='TRAIN only',folds=5,seed=0,settings=85,
            objective='mean(affine,feature_dropout,noise)',clean_reference='exact control',
            clean_tolerance=.0025,per_diagnostic_tolerance=.0025,minimum_diagnostic_gain=.001,
            minimum_historical_joint_gain=.0005,minimum_improved_folds=3,
            paired_fold_score='mean(affine,feature_dropout,noise)',tie_break='fewer trees, then earlier setting',
            new_cv_fitting=False,staff_c2_reconstruction=False,runner_up_rescue=False))


def verify_all_scenarios(setting,context):
    records=verify_native(setting,context)
    for record,(_,va) in zip(records,grouped_kfold(context['qids'],k=5,seed=0)):
        fold=record['fold'];sources=setting['sources'];counts=[s[2] for s in sources]
        models=[lgb.Booster(model_file=str(Path(context['directories'][v])/f'candidate_{c}_fold_{fold}.txt'))
                for v,c,_ in sources]
        native=models[0] if len(models)==1 else combine(models,setting['weights'],counts)
        width=context['prepared'].shape[1]//3;raw=context['prepared'][va,:width]
        checks=[dict(scenario='clean',maximum_absolute_error=record['native_max_abs_error'],
                     native_ndcg_10=record['native_ndcg_10'],explicit_ndcg_10=record['explicit_ndcg_10'])]
        for index,name in enumerate(SCENARIOS[1:],1):
            perturbed=perturb_features(raw,context['qids'][va],name)
            actual=native.predict(features_for_model(native,perturbed,context['qids'][va]),
                                  num_iteration=counts[0] if len(models)==1 else None,num_threads=4)
            expected=sum(w*context['predictions'][s][index,va] for s,w in zip(sources,setting['weights']))
            error=float(np.max(np.abs(actual-expected)))
            actual_ndcg=mean_ndcg_from_scores(actual,context['labels'][va],context['qids'][va])
            expected_ndcg=mean_ndcg_from_scores(expected,context['labels'][va],context['qids'][va])
            if error>5e-12 or abs(actual_ndcg-expected_ndcg)>1e-12:
                raise ValueError('native diagnostic average differs')
            checks.append(dict(scenario=name,maximum_absolute_error=error,native_ndcg_10=actual_ndcg,
                               explicit_ndcg_10=expected_ndcg))
        record['all_scenarios']=checks
    return records


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--train',default='data/train/train.txt')
    parser.add_argument('--baseline',default='results/lambdamart_cv_release_v12.json')
    parser.add_argument('--out',default='results/lambdamart_cv_v13.json')
    versions=('v3','v4','v5','v6','v7','v8','v11','v12')
    for v in versions:
        parser.add_argument('--'+v+'-dir',default='runs/cv_'+v)
    args=parser.parse_args();started=time.perf_counter();settings=specifications()
    directories={v:getattr(args,v+'_dir') for v in versions}
    rows,context=measure(args.train,directories,settings)
    result=select(rows,json.loads(Path(args.baseline).read_text()))
    chosen=next(s for s in settings if s['name']==result['selected']['name'])
    result.update(arguments=vars(args),rows=context['rows'],queries=context['queries'],
                  native_equivalence=verify_all_scenarios(chosen,context),seconds=time.perf_counter()-started)
    target=Path(args.out);target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_suffix('.pending.json');temporary.write_text(json.dumps(result,indent=2)+'\n');temporary.replace(target)
    print(json.dumps(dict(promoted=result['promoted'],selected=result['selected']['name'],gates=result['promotion_gates'])),flush=True)


if __name__=='__main__':
    main()
