"""One-time V14 launch with exclusive metadata and atomic scheduler receipts."""
import fcntl,hashlib,json,re,subprocess
from pathlib import Path

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
lock=Path('runs/v14_launch.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
p=Path('runs/v14_jobs.json');assert not p.exists(),'inspect existing actual IDs; never launch twice'
m=json.loads(Path('runs/v14_intent.json').read_text())
assert m['choice_pushed'] and re.fullmatch('[a-f0-9]{40}',m['choice_commit'])
for group in ('source_sha256','configuration_sha256','helper_sha256'):
 for name,h in m[group].items():assert digest(name)==h,name
for name,h in m['baseline_hashes'].items():assert digest('submission/'+name)==h,name
assert digest('results/neural_history.json')==m['baseline_neural_history_sha256']
assert not Path('results/dev_metrics_v14.json').exists()
for split,path in [('TRAIN','data/train/train.txt'),('DEV','data/dev/vali.txt')]:assert digest(path)==m['fit_input_sha256'][split]
with p.open('x') as f:f.write(json.dumps(m,indent=2)+'\n')
def save():
 t=p.with_suffix('.tmp');t.write_text(json.dumps(m,indent=2)+'\n');t.replace(p)
for key,wrapper in [('new_group_primary','runs/hpc_new_primary_v14.pbs'),('new_group_independent','runs/hpc_new_independent_v14.pbs'),('native_cv','runs/hpc_native_cv_v14.pbs')]:
 subprocess.run(['bash','-n',wrapper],check=True)
 m['pending_submission']=dict(key=key,command=['qsub',wrapper]);save()
 job=subprocess.check_output(m['pending_submission']['command'],text=True).strip()
 assert re.fullmatch(r'\d+\.pbshpc',job),'reconcile unexpected scheduler response'
 m[key]=job;m.pop('pending_submission');save();print(key,job,flush=True)
m['first_stage_submitted']=True;save()
