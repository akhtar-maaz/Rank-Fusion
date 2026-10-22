"""Verify frozen choices and independent fits before installing either artifact.

Operational helper only: no fitting, selection, or DEV evaluation.
"""
import hashlib
import json
from pathlib import Path
import shutil


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    target = Path(path)
    temporary = target.with_suffix(target.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(target)


def verify():
    metadata = read('runs/v12_jobs.json')
    assert metadata.get('selection_commit') and not metadata.get('pending_submission')
    audit = read('results/v12_selection_audit.json')
    assert digest('results/v12_selection_audit.json') == metadata['selection_audit_sha256']
    for path, expected in audit['selection_sha256'].items():
        assert digest(path) == expected, path
    for path, expected in metadata['source_sha256'].items():
        assert digest(path) == expected, path
    for path, expected in metadata['completion_helper_sha256'].items():
        assert digest(path) == expected, path
    for name, expected in metadata['baseline_hashes'].items():
        assert digest('submission/' + name) == expected, name
    assert Path('results/neural_history.json').read_bytes() == Path('results/neural_history_v11.json').read_bytes()
    assert not Path('results/dev_metrics_v12.json').exists(), 'final DEV evaluation already exists; do not repeat'
    tree = read('results/lambdamart_cv_v12.json')
    neural = read('results/neural_selection_v12.json')
    assert tree['selected'] == audit['tree_selected'] and tree['promoted'] == audit['tree_promoted']
    assert neural['selected']['candidate'] == audit['neural_selected'] and neural['promoted'] == audit['neural_promoted']
    assert bool(metadata['tree_refit_required']) == tree['promoted']
    assert bool(metadata['pairwise_refit_required']) == neural['promoted']
    # All branch checks precede any submission mutation.
    if tree['promoted']:
        config = read('configs/v12_tree.json')
        assert config['selected'] == tree['selected']
        first = read('runs/refit_v12/tree_primary/fit.json')
        second = read('runs/refit_v12/tree_independent/fit.json')
        for record in (first, second):
            assert record['fit_splits'] == ['TRAIN', 'DEV'] and record['use_dev_for_fitting']
            assert record['query_ids_namespaced_by_split'] and record['dev_tree_metric_is_in_sample']
            assert not record['early_stopping_used']
            assert (record['rows'], record['queries'], record['raw_features']) == (958671, 8000, 136)
            assert record['actual_trees'] == tree['selected']['trees'] and record['actual_features'] == 408
            assert record['selected'] == config['selected']
            assert record['configuration_sha256'] == metadata['final_configuration_sha256'] == digest('configs/v12_tree.json')
            assert record['selection_provenance'] == config['selection_provenance']
            assert {entry['split']: entry['sha256'] for entry in record['inputs']} == metadata['fit_input_sha256']
            for path, expected in record['source_sha256'].items():
                assert digest(path) == expected == metadata['source_sha256'][path], path
            assert record['python'] == '3.9.21'
        for key in ('inputs', 'source_sha256', 'selected', 'configuration_sha256', 'versions', 'python'):
            assert first[key] == second[key], key
        tree_hash = digest('runs/refit_v12/tree_primary/model.txt')
        assert tree_hash == digest('runs/refit_v12/tree_independent/model.txt') == first['artifact_sha256'] == second['artifact_sha256']
    else:
        tree_hash = metadata['baseline_hashes']['model.txt']
        assert read('results/tree_fit_v10.json')['artifact_sha256'] == tree_hash

    if neural['promoted']:
        primary = read('runs/refit_v12/pairwise_primary/fit.json')
        independent = read('runs/refit_v12/pairwise_independent/fit.json')
        receipts = [read('runs/refit_v12/' + branch + '/runtime.json') for branch in ('pairwise_primary', 'pairwise_independent')]
        assert receipts[0] == receipts[1] and receipts[0]['python'] == '3.9.21'
        assert all(receipts[0]['environment'][name] == '4' for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'))
        for record in (primary, independent):
            assert record['promoted'] and record['trained_on'] == 'TRAIN only'
            assert record['candidate'] == neural['selected']['candidate']
            assert record['arguments']['fresh'] and record['arguments']['selection'] == 'results/neural_selection_v12.json'
            assert record['source_sha256'] == neural['context']['source_sha256']
            assert record['runtime'] == neural['context']['runtime']
            assert record['standardizers_identical'] and record['exported_config'] == {'n_features': 136, 'hidden': [256, 128]}
            assert record['compilation']['score_atol'] == 2e-6 and record['compilation']['score_rtol'] == 2e-5
            assert len(record['members']) == 2
            for member, spec in zip(record['members'], record['candidate']['members']):
                assert member['specification'] == spec
                context = member['identity']['context']
                assert (context['rows'], context['queries'], context['features']) == (723412, 6000, 136)
                assert context['train_sha256'] == metadata['fit_input_sha256']['TRAIN'] and context['split'] == 'full TRAIN'
                assert context['source_sha256'] == record['source_sha256'] and context['runtime'] == record['runtime']
                assert digest(member['artifact']) == member['artifact_sha256']
        for key in ('candidate', 'source_sha256', 'runtime', 'exported_config'):
            assert primary[key] == independent[key], key
        for a, b in zip(primary['members'], independent['members']):
            for key in ('specification', 'identity', 'cache_key', 'history', 'artifact_sha256', 'state_sha256', 'standardizer_sha256'):
                assert a[key] == b[key], key
        pairwise_hash = digest('runs/refit_v12/pairwise_primary/pairwise_model.pt')
        assert pairwise_hash == digest('runs/refit_v12/pairwise_independent/pairwise_model.pt') == primary['artifact_sha256'] == independent['artifact_sha256']
    else:
        pairwise_hash = metadata['baseline_hashes']['pairwise_model.pt']

    if tree['promoted']:
        shutil.copyfile('runs/refit_v12/tree_primary/model.txt', 'submission/model.txt')
        first.update(shipped_artifact='submission/model.txt', shipped_sha256=tree_hash)
        write('results/tree_fit_v12.json', first)
        write('results/tree_fit_reproduced_v12.json', second)
        write('results/lambdamart_reproducibility.json', dict(
            artifact='submission/model.txt', original_sha256=tree_hash, reproduced_sha256=tree_hash,
            byte_identical=True, refitted=True, fit_splits=['TRAIN', 'DEV'],
            primary_fit_record='results/tree_fit_v12.json', independent_fit_record='results/tree_fit_reproduced_v12.json',
            configuration='configs/v12_tree.json', configuration_files_byte_identical=True))
        provenance = dict(protocol='docs/EXPERIMENTS_V12.md', fit_splits=['TRAIN', 'DEV'],
                          refitted_in_v12=True, fit_record='results/tree_fit_v12.json',
                          configuration='configs/v12_tree.json', cv_scope='V12 released TRAIN only')
    else:
        provenance = dict(protocol='docs/EXPERIMENTS_V12.md', fit_splits=['TRAIN', 'DEV'],
                          refitted_in_v12=False, retained_from='V10', fit_record='results/tree_fit_v10.json',
                          configuration='configs/v10_tree.json', cv_scope='V12 TRAIN comparison retained V10',
                          original_configuration_selection='historical TRAIN-only CV')
    tree['final_refit'] = provenance
    write('results/lambdamart_cv.json', tree)
    if neural['promoted']:
        shutil.copyfile('runs/refit_v12/pairwise_primary/pairwise_model.pt', 'submission/pairwise_model.pt')
        write('results/pairwise_refit_v12.json', primary)
        write('results/pairwise_refit_reproduced_v12.json', independent)
        write('results/pairwise_reproducibility_v12.json', dict(
            artifact='submission/pairwise_model.pt', original_sha256=pairwise_hash, reproduced_sha256=pairwise_hash,
            byte_identical=True, refitted=True, fit_splits=['TRAIN'], runtime_receipt=receipts[0],
            primary_fit_record='results/pairwise_refit_v12.json', independent_fit_record='results/pairwise_refit_reproduced_v12.json',
            input_source_configuration_runtime_equal=True, member_states_statistics_histories_equal=True))
        history = read('results/neural_history_v11.json')
        history['model_arguments']['pairwise'] = primary['candidate']
        history['selected_on']['pairwise'] = 'V12 five-fold fixed-epoch TRAIN assessment; exported plain-MLP seed average'
        history['models']['pairwise'] = dict(candidate=primary['candidate'], full_train_members=primary['members'],
            compilation=primary['compilation'], exported_config=primary['exported_config'], fixed_epochs=True,
            export_kind='compiled arithmetic average', selection_record='results/neural_selection_v12.json')
        write('results/neural_history.json', history)
    for name in ('pointwise_model.pt', 'fusion_config.json', 'recipe.py', 'pointwise_pairwise.py'):
        assert digest('submission/' + name) == metadata['baseline_hashes'][name]
    result = dict(tree_refitted=tree['promoted'], pairwise_refitted=neural['promoted'],
                  tree_sha256=tree_hash, pairwise_sha256=pairwise_hash,
                  selection_commit=metadata['selection_commit'], final_tree_provenance=provenance,
                  pointwise_fusion_core_and_recipe_preserved=True,
                  independent_byte_identity_required_for_new_artifacts=True)
    write('results/verification_v12.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    verify()
