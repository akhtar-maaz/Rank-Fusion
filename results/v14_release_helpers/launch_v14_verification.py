"""Append one dependent final verification; never replace existing fit IDs."""
import fcntl,hashlib,json,re,subprocess
from pathlib import Path
lock=Path('runs/v14_launch.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
p=Path('runs/v14_jobs.json');m=json.loads(p.read_text());intent=json.loads(Path('runs/v14_verification_intent.json').read_text())
assert m['first_stage_submitted'] and not m.get('pending_submission') and 'verification' not in m
assert intent['choice_commit']==m['choice_commit'] and not Path('results/dev_metrics_v14.json').exists()
for name,h in intent['helper_sha256'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h,name
m['verification_helper_sha256']=intent['helper_sha256']
command=['qsub','-W','depend=afterok:'+':'.join(m[k] for k in ('new_group_primary','new_group_independent','native_cv')),'runs/hpc_verify_v14.pbs']
subprocess.run(['bash','-n',command[-1]],check=True)
def save():
 t=p.with_suffix('.tmp');t.write_text(json.dumps(m,indent=2)+'\n');t.replace(p)
m['pending_submission']=dict(key='verification',command=command);save()
job=subprocess.check_output(command,text=True).strip();assert re.fullmatch(r'\d+\.pbshpc',job)
m['verification']=job;m.pop('pending_submission');save();print(job,flush=True)
