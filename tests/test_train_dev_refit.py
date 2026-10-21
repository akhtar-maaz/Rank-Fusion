import hashlib
import json
import sys

import numpy as np
import pytest

from scripts import train_model
from scripts.lambdamart_utils import default_params


def write_split(path, seed, queries):
    rng = np.random.RandomState(seed)
    lines = []
    for query in queries:
        for _ in range(12):
            features = rng.normal(size=3)
            label = int(features[0] > 0) + int(features[1] > 0)
            lines.append(f"{label} qid:{query} " + " ".join(f"{i+1}:{v:.7g}" for i, v in enumerate(features)))
    path.write_text("\n".join(lines) + "\n")


@pytest.mark.parametrize('include_dev', [False, True])
def test_fit_record_proves_actual_split_scope_and_fixed_config(tmp_path, monkeypatch, include_dev):
    train, dev = tmp_path / 'train.txt', tmp_path / 'vali.txt'
    write_split(train, 1, ['a', 'b', 'c', 'd'])
    write_split(dev, 2, ['a', 'e'])  # Intentional cross-split query-ID reuse.
    config = tmp_path / 'config.json'
    selected = dict(params=default_params(7, min_data_in_leaf=2), trees=8, feature_mode='query_all')
    provenance = dict(selection_split='TRAIN only', new_hyperparameter_search=False,
                      final_fit_splits=['TRAIN', 'DEV'])
    config.write_text(json.dumps(dict(selected=selected, selection_provenance=provenance)))
    model, record = tmp_path / 'model.txt', tmp_path / 'fit.json'
    args = ['train_model.py', '--train', str(train), '--dev', str(dev), '--config', str(config),
            '--out', str(model), '--fit-record', str(record)]
    if include_dev:
        args.append('--use-dev')
    monkeypatch.setattr(sys, 'argv', args)
    train_model.main()
    actual = json.loads(record.read_text())
    assert actual['selected'] == selected and actual['selection_provenance'] == provenance
    assert actual['actual_trees'] == 8 and actual['actual_features'] == 9
    assert actual['configuration_sha256'] == hashlib.sha256(config.read_bytes()).hexdigest()
    assert actual['artifact_sha256'] == hashlib.sha256(model.read_bytes()).hexdigest()
    assert actual['fit_splits'] == (['TRAIN', 'DEV'] if include_dev else ['TRAIN'])
    assert actual['queries'] == (6 if include_dev else 4)
    assert actual['rows'] == (72 if include_dev else 48)
    assert actual['use_dev_for_fitting'] == actual['query_ids_namespaced_by_split'] == include_dev
    assert actual['dev_tree_metric_is_in_sample'] == include_dev
    assert not actual['early_stopping_used']
    assert all(actual['source_sha256'].values())
    assert [entry['queries'] for entry in actual['inputs']] == ([4, 2] if include_dev else [4])


def test_use_dev_requires_an_actual_dev_input(tmp_path, monkeypatch):
    train = tmp_path / 'train.txt'
    write_split(train, 4, ['a', 'b'])
    monkeypatch.setattr(sys, 'argv', ['train_model.py', '--train', str(train), '--use-dev'])
    with pytest.raises(SystemExit, match='needs a dev file'):
        train_model.main()


def test_v10_freezes_evaluated_tree_parameters_without_dev_search():
    from pathlib import Path
    root = Path(__file__).parents[1]
    config = json.loads((root / 'configs/v10_tree.json').read_text())
    source = root / config['selection_provenance']['source']
    assert config['selected'] == json.loads(source.read_text())['selected']
    assert config['selection_provenance']['source_sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert config['selection_provenance']['selection_split'] == 'TRAIN only'
    assert not config['selection_provenance']['new_hyperparameter_search']
    assert config['selected']['trees'] == 3800
