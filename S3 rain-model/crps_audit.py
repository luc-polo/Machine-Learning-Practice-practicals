"""Audit the original CRPS implementation and replay the saved submission.

Run from the repository root:
    .venv/bin/python "S3 rain-model/crps_audit.py"

This reads the existing notebook definitions and saved model. It does not
train, upload, deploy, or execute the notebook's setup/submission cells.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parent
path = root/'mlp_s3_rain_model.ipynb'
nb = json.loads(path.read_text())
ns = dict(np=np, pd=pd, FILE=str(root/'weather_europe_2020_2026.csv.gz'))
for cell in [6, 8, 10, 12]:
    exec(''.join(nb['cells'][cell]['source']), ns)
exec(''.join(nb['cells'][14]['source']).split('runs_clim =')[0], ns)
crps = ns['crps']
rng = np.random.default_rng(18)
maximum = 0.0
for m in [1, 2, 3, 30, 50, 100]:
    members = rng.normal(0.5, 3, (37, 2, m))
    y = rng.exponential(0.3, (37, 2))
    x = np.maximum(members, 0)
    brute = np.abs(x-y[...,None]).mean(-1) - .5*np.abs(x[..., :, None]-x[..., None, :]).mean((-1,-2))
    maximum = max(maximum, float(np.abs(crps(members,y)-brute).max()))
print('Maximum CRPS error vs pairwise definition:',maximum, flush=True)
spec=importlib.util.spec_from_file_location('submitted_s3_agent',root/'rain_submission/agent.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
agent=module.Agent()
clim=ns['Climatology'](ns['SPLIT'])
issues=ns['TEST']
all_losses=[]
for start in range(0,len(issues),50):
    batch=issues[start:start+50]
    features=pd.concat([module.rain_features(ns['window'](i),ns['hours'][i]) for i in batch],ignore_index=True)
    members=agent.members(features).reshape(len(batch),len(ns['PANEL']),2,100)
    y=np.stack([ns['rain'][batch+h] for h in ns['HORIZONS']],axis=-1)
    loss=crps(members,y).mean(axis=1)
    all_losses.append(loss)
    if start==0:
        request=ns['make_request'](batch[0])
        direct=ns['check'](agent.predict(request),request)
        np.testing.assert_allclose(direct,members[0])
loss=np.concatenate(all_losses)
times=ns['hours'][issues]
runs=pd.DataFrame(loss,index=times,columns=['6h','48h'])
runs['score']=runs.mean(axis=1)
print('EXPORTED MODEL HELD-OUT REPLAY:',runs.mean().to_dict(),flush=True)
print('Monthly replay:',runs.groupby(runs.index.month).mean().to_string(),flush=True)
clim_loss=[]
for i in issues:
    request=ns['make_request'](i)
    forecast=np.asarray(clim.predict(request)['rain'])
    y=np.stack([ns['rain'][i+h] for h in ns['HORIZONS']],axis=-1)
    clim_loss.append(crps(forecast,y).mean(axis=0))
clim_runs=pd.DataFrame(clim_loss,index=times,columns=['6h','48h'])
clim_runs['score']=clim_runs.mean(axis=1)
print('CLIMATOLOGY REPLAY:',clim_runs.mean().to_dict(),flush=True)
score=.19889557444444445
print('Historical single runs at least as bad:',int((runs.score>=score).sum()),'/',len(runs),flush=True)
print('Historical max / score quantiles:',runs.score.max(),runs.score.quantile([.5,.9,.95,.99]).to_dict(),flush=True)
paired=runs.score-clim_runs.score
print('Single-run model worse than climatology:',float((paired>0).mean()),flush=True)
out=root/'crps_audit_results'
out.mkdir(exist_ok=True)
runs.to_csv(out/'exported_model_replay.csv')
clim_runs.to_csv(out/'climatology_replay.csv')
summary={'crps_max_abs_error':maximum,'replay':runs.mean().to_dict(),'climatology':clim_runs.mean().to_dict(),'n_runs':len(runs),'historical_runs_ge_live_score':int((runs.score>=score).sum())}
(out/'summary.json').write_text(json.dumps(summary,indent=2))
for name in ['agent.py','model.joblib']:
    actual=hashlib.sha256((root/'rain_submission'/name).read_bytes()).hexdigest()
    expected=json.loads((root/'rain_submission/submission_receipt.json').read_text())['files'][name]
    print('Receipt checksum',name,actual==expected,flush=True)
