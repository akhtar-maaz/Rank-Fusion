"""Append one recovery verification, preserving all original successful/failed attempts."""
import fcntl,hashlib,json,re,subprocess
from pathlib import Path
lock=Path('runs/v14_launch.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
p=Path('runs/v14_jobs.json');m=json.loads(p.read_text());r=json.loads(Path('runs/v14_recovery_intent.json').read_text())
assert not m.get('pending_submission') and 'recovery_1' not in m and r['choice_pushed']
assert not Path('results/dev_metrics_v14.json').exists()
meta=json.loads(Path('runs/v14_job_metadata.json').read_text())
for k in ('new_group_primary','new_group_independent','native_cv'):assert meta[m[k]]['job_state']=='F' and meta[m[k]]['Exit_status']=='0'
assert meta[m['verification']]['job_state']=='F' and meta[m['verification']]['Exit_status']=='1'
for group in ('configuration_sha256','helper_sha256'):
 for name,h in r[group].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h,name
m['recovery_1']=r
command=['qsub','runs/hpc_verify_v14_recovery.pbs'];subprocess.run(['bash','-n',command[-1]],check=True)
def save():
 t=p.with_suffix('.tmp');t.write_text(json.dumps(m,indent=2)+'\n');t.replace(p)
m['pending_submission']=dict(key='recovery_1.verification',command=command);save()
job=subprocess.check_output(command,text=True).strip();assert re.fullmatch(r'\d+\.pbshpc',job)
m['recovery_1']['verification']=job;m.pop('pending_submission');save();print(job,flush=True)
