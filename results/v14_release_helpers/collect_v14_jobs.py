"""Preserve complete actual V13 scheduler records, including after expiry."""
import json
from pathlib import Path
import re
import subprocess

m=json.loads(Path('runs/v14_jobs.json').read_text())
jobs=[m[key] for key in ('new_group_primary','new_group_independent','native_cv','verification') if key in m]
assert len(jobs)==len(set(jobs))
path=Path('runs/v14_job_metadata.json')
records=json.loads(path.read_text()) if path.exists() else {}
for job in jobs:
    try:r=subprocess.run(['qstat','-xf',job],capture_output=True,text=True,timeout=20)
    except subprocess.TimeoutExpired:
        if records.get(job,{}).get('job_state')!='F':
            records[job]=dict(records.get(job,{}),query_error='qstat exceeded20seconds')
        continue
    if r.returncode:
        if records.get(job,{}).get('job_state')!='F':
            records[job]=dict(records.get(job,{}),query_error=r.stderr.strip())
        continue
    row={}
    for line in r.stdout.splitlines():
        found=re.match(r'\s+([\w.]+) = (.*)$',line)
        if found and (found[1] in ('Job_Name','job_state','exec_host','Exit_status','stime','obittime','comment') or
                      found[1].startswith(('resources_used.','Resource_List.'))):row[found[1]]=found[2]
    if row.get('job_state'):
        prior={k:v for k,v in records.get(job,{}).items() if k!='query_error'}
        if prior.get('job_state')!='F' or row['job_state']=='F':records[job]=dict(prior,**row)
temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(records,indent=2)+'\n');temporary.replace(path)
print(json.dumps({j:{k:v for k,v in records[j].items() if k in
    ('job_state','Exit_status','resources_used.walltime','resources_used.mem','query_error','comment')} for j in jobs},indent=2))
