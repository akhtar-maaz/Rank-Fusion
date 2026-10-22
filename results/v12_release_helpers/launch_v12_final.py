"""Submit final fits once, only after the pushed TRAIN-only selection audit.

An unresolved pre-qsub intent must be reconciled manually, never retried.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


path = Path('runs/v12_jobs.json')
lock = Path('runs/v12_final_launch.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
metadata = json.loads(path.read_text())
assert not metadata.get('pending_submission'), 'reconcile pending scheduler intent before any retry'
intent = json.loads(Path('runs/v12_final_intent.json').read_text())
assert intent['selection_pushed'] and re.fullmatch(r'[0-9a-f]{40}', intent['selection_commit'])
assert all(metadata[key] == intent['first_stage_ids'][key] for key in ('cv_array', 'neural_selection', 'tree_selection'))
audit = json.loads(Path('results/v12_selection_audit.json').read_text())
assert digest('results/v12_selection_audit.json') == intent['selection_audit_sha256']
assert audit['tree_promoted'] == intent['tree_refit_required']
assert audit['neural_promoted'] == intent['pairwise_refit_required']
for name, expected in audit['selection_sha256'].items():
    assert digest(name) == expected, name
for name, expected in metadata['source_sha256'].items():
    assert digest(name) == expected, name
for name, expected in metadata['baseline_hashes'].items():
    assert digest('submission/' + name) == expected, name
assert not Path('results/dev_metrics_v12.json').exists()
for name, expected in intent['completion_helper_sha256'].items():
    assert digest(name) == expected, name
for branch in ('tree_primary', 'tree_independent', 'pairwise_primary', 'pairwise_independent'):
    assert not Path('runs/refit_v12/' + branch + '/fit.json').exists(), 'unexpected completed fit: ' + branch
if intent['tree_refit_required']:
    config = json.loads(Path('configs/v12_tree.json').read_text())
    assert config['selected'] == audit['tree_selected']
    assert digest('configs/v12_tree.json') == intent['final_configuration_sha256']
for key in ('selection_commit', 'selection_audit_sha256', 'tree_refit_required', 'pairwise_refit_required',
            'completion_helper_sha256', 'final_configuration_sha256'):
    if key in metadata:
        assert metadata[key] == intent[key], 'do not replace existing intent: ' + key
    metadata[key] = intent[key]


def save():
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(metadata, indent=2) + '\n')
    temporary.replace(path)


save()


def submit(key, wrapper, dependencies=()):
    if key in metadata:
        print(key, metadata[key], 'already recorded', flush=True)
        return metadata[key]
    subprocess.run(['bash', '-n', wrapper], check=True)
    command = ['qsub']
    if dependencies:
        command += ['-W', 'depend=afterok:' + ':'.join(dependencies)]
    command.append(wrapper)
    metadata['pending_submission'] = dict(key=key, command=command)
    save()
    job = subprocess.check_output(command, text=True).strip()
    if not re.fullmatch(r'\d+\.pbshpc', job):
        raise RuntimeError('unexpected qsub response; reconcile intent: ' + job)
    metadata[key] = job
    metadata.pop('pending_submission')
    save()
    print(key, job, flush=True)
    return job


dependencies = []
if intent['tree_refit_required']:
    dependencies += [submit('tree_final_train', 'runs/hpc_tree_primary_v12.pbs'),
                     submit('tree_independent_reproduction', 'runs/hpc_tree_independent_v12.pbs')]
if intent['pairwise_refit_required']:
    dependencies += [submit('pairwise_final_train', 'runs/hpc_pairwise_primary_v12.pbs'),
                     submit('pairwise_independent_reproduction', 'runs/hpc_pairwise_independent_v12.pbs')]
submit('verification', 'runs/hpc_verify_v12.pbs', dependencies)
metadata['final_stage_submitted'] = True
save()
