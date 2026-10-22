"""Submit the audited V13 final branch once, preserving each scheduler intent."""
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


lock=Path('runs/v13_final_launch.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
path=Path('runs/v13_jobs.json');m=json.loads(path.read_text())
assert not m.get('pending_submission'),'reconcile unresolved intent; never submit duplicate jobs'
intent=json.loads(Path('runs/v13_final_intent.json').read_text())
assert intent['selection_pushed'] and re.fullmatch(r'[0-9a-f]{40}',intent['selection_commit'])
assert m['selection']==intent['selection_job']
assert digest('results/v13_selection_audit.json')==intent['selection_audit_sha256']
audit=json.loads(Path('results/v13_selection_audit.json').read_text())
assert digest('results/lambdamart_cv_v13.json')==audit['selection_sha256']
assert audit['promoted']==intent['tree_refit_required']
for p,h in m['source_sha256'].items():assert digest(p)==h,p
for name,h in m['baseline_hashes'].items():assert digest('submission/'+name)==h,name
assert digest('results/neural_history.json')==m['baseline_neural_history_sha256']
for p,h in intent['completion_helper_sha256'].items():assert digest(p)==h,p
assert not Path('results/dev_metrics_v13.json').exists()
if intent['tree_refit_required']:
    assert digest('configs/v13_tree.json')==intent['final_configuration_sha256']
    assert json.loads(Path('configs/v13_tree.json').read_text())['selected']==audit['selected']
for key in ('tree_final_train','tree_independent_reproduction','verification'):
    assert key not in m,'final stage already initialized; inspect existing job: '+key
for branch in ('primary','independent'):
    assert not Path('runs/refit_v13/'+branch+'/fit.json').exists()
for key in ('selection_commit','selection_audit_sha256','tree_refit_required','final_configuration_sha256','completion_helper_sha256'):
    if key in m:assert m[key]==intent[key]
    m[key]=intent[key]


def save():
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(m,indent=2)+'\n');temporary.replace(path)


def submit(key,wrapper,dependencies=()):
    command=['qsub']
    if dependencies:command+=['-W','depend=afterok:'+':'.join(dependencies)]
    command.append(wrapper);subprocess.run(['bash','-n',wrapper],check=True)
    m['pending_submission']=dict(key=key,command=command);save()
    job=subprocess.check_output(command,text=True).strip()
    if not re.fullmatch(r'\d+\.pbshpc',job):raise RuntimeError('reconcile unexpected qsub response: '+job)
    m[key]=job;m.pop('pending_submission');save();print(key,job,flush=True)
    return job


save();dependencies=[]
if intent['tree_refit_required']:
    dependencies=[submit('tree_final_train','runs/hpc_tree_primary_v13.pbs'),
                  submit('tree_independent_reproduction','runs/hpc_tree_independent_v13.pbs')]
submit('verification','runs/hpc_verify_v13.pbs',dependencies)
m['final_stage_submitted']=True;save()
