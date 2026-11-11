import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import eval_slices  # register project/RecBole model classes before reading fixtures
from ikgr_core.pause_resume import Control, RunLock


def worker(folder, model, *args, expected=0):
    env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8',PYTHONHASHSEED='2020',WANDB_MODE='disabled')
    r=subprocess.run([sys.executable,str(ROOT/'tests/pause_worker.py'),'--folder',str(folder),
                      '--model',model,*args],env=env,cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
    assert r.returncode==expected,r.stdout+r.stderr
    return r


def equal(a,b):
    if torch.is_tensor(a): assert torch.equal(a.cpu(),b.cpu())
    elif isinstance(a,np.ndarray): assert np.array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a: equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b): equal(x,y)
    else: assert a==b


@pytest.mark.parametrize('model,pause_epoch',[('BPR',0),('BPR',1),('BPR',5),('IKGR',0),('IKGR',1)])
def test_pause_matches_uninterrupted_across_processes(tmp_path,model,pause_epoch):
    baseline=tmp_path/'baseline'
    paused=tmp_path/'paused'
    worker(baseline,model,'--plain')
    worker(paused,model,'--pause-epoch',str(pause_epoch),expected=75)
    assert (paused/'resume.pt').exists() and not (paused/'final.pt').exists()
    state=torch.load(paused/'resume.pt',map_location='cpu')
    assert state['last_epoch']==pause_epoch
    assert state['fit_complete']==(pause_epoch==5)
    # A second pause after resuming tests repeated snapshots and the best-model bundle.
    if pause_epoch==0:
        worker(paused,model,'--resume','--pause-epoch','2',expected=75)
    worker(paused,model,'--resume')
    a=torch.load(baseline/'final.pt',map_location='cpu')
    b=torch.load(paused/'final.pt',map_location='cpu')
    equal(a,b)
    assert a['best_epoch']==1 and a['fit_info']['last_epoch']==5 and a['cur_step']==2


def test_mismatch_refused_without_overwriting_snapshot(tmp_path):
    worker(tmp_path,'BPR','--pause-epoch','1',expected=75)
    before=hashlib.sha256((tmp_path/'resume.pt').read_bytes()).hexdigest()
    r=worker(tmp_path,'BPR','--resume','--signature','changed',expected=1)
    assert 'refusing unsafe resume' in r.stderr
    assert hashlib.sha256((tmp_path/'resume.pt').read_bytes()).hexdigest()==before


def test_control_status_and_exclusive_lock(tmp_path):
    control=Control(tmp_path)
    assert control.read()=={'state':'idle','pause_requested':False}
    control.ask_pause()
    assert control.read()['pause_requested']
    with RunLock(tmp_path):
        with pytest.raises(RuntimeError,match='already running'):
            with RunLock(tmp_path): pass
    with RunLock(tmp_path): pass


def test_launcher_marks_gpu_release_only_after_child_exit(tmp_path,monkeypatch):
    import yaml
    from types import SimpleNamespace
    from scripts import run_todo_5_8 as launch
    cfg=tmp_path/'config.yaml'
    cfg.write_text(yaml.safe_dump({'paths':{'workdir':str(tmp_path/'run')},'recbole':{'gpu_id':'0'}}))
    control=Control(tmp_path/'run/control')
    calls=[]
    def child(cmd,**kw):
        calls.append(cmd)
        if '--check' in cmd:
            assert control.read()['gpu_released'] is False
            return SimpleNamespace(returncode=0)
        assert cmd[-1]=='eval_slices.py'
        control.status('checkpoint_saved',spec='BPR',seed=2020,last_epoch=0)
        assert control.read().get('gpu_released') is not True
        return SimpleNamespace(returncode=75)
    monkeypatch.setattr(launch.subprocess,'run',child)
    monkeypatch.setattr(sys,'argv',['run_todo_5_8.py','--run','--phase','5','--config',str(cfg)])
    previous_env=dict(os.environ)
    previous_cwd=os.getcwd()
    try:
        launch.main()
    finally:
        os.environ.clear()
        os.environ.update(previous_env)
        os.chdir(previous_cwd)
    status=control.read()
    assert len(calls)==2 and status['state']=='paused' and status['gpu_released'] is True
    assert status['phase']=='5' and status['last_epoch']==0


def test_eval_queue_pause_resume_and_result_commit(tmp_path):
    import yaml
    # Reuse the small deterministic atomic-data fixture; all work stays under tmp_path.
    worker(tmp_path/'data','BPR','--plain')
    output=tmp_path/'output'
    config=tmp_path/'config.yaml'
    config.write_text(yaml.safe_dump({'paths':{'inter_file':str(tmp_path/'data/input.csv'),'workdir':str(output)},
        'recbole':{'dataset':'tiny','embedding_size':16,'epochs':4,'eval_step':2,'stopping_step':3,
                   'metrics':['NDCG','Recall'],'topk':[10,30],'gpu_id':'','worker':0,'train_batch_size':64,
                   'eval_users_per_batch':8}}))
    control=Control(output/'control')
    env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8',PYTHONHASHSEED='2020',CUDA_VISIBLE_DEVICES='',
             IKGR_CONFIG=str(config),IKGR_SPLIT='TO',IKGR_SEEDS='2020',IKGR_SPECS='BPR',IKGR_EPOCHS='4',
             IKGR_EVAL_STEP='2',IKGR_STOPPING_STEP='3',IKGR_VALID_METRIC='NDCG@10',IKGR_DUMP_VALID='1',
             IKGR_CONTROL_DIR=str(control.folder),IKGR_POP_RERANK_GRID='0.25',IKGR_POP_RERANK_SPECS='BPR',
             TEST_PAUSE_AFTER_FIRST='1',IKGR_RESUME_ALLOWED='0',IKGR_REQUIRE_PROVENANCE='0')
    def run(expected):
        p=subprocess.run([sys.executable,str(ROOT/'tests/pause_queue_worker.py')],cwd=ROOT,env=env,
                         capture_output=True,text=True,encoding='utf-8')
        assert p.returncode==expected,p.stdout+p.stderr
        return p
    run(75)
    assert not (output/'slice_eval_TO_result.json').exists()
    assert (output/'resume/BPR__s2020.pt').exists()
    # The actual launcher pause/status commands must work without importing torch/CUDA.
    p=subprocess.run([sys.executable,str(ROOT/'scripts/run_todo_5_8.py'),'--status','--config',str(config)],
                     capture_output=True,text=True,encoding='utf-8')
    assert p.returncode==0,p.stderr
    assert json.loads(p.stdout)['pause_requested']
    control.request.unlink()
    env.update(TEST_PAUSE_AFTER_FIRST='0',IKGR_RESUME_ALLOWED='1')
    run(0)
    result=json.loads((output/'slice_eval_TO_result.json').read_text())
    assert set(result)=={'BPR','BPR_pop0p25'}
    assert not (output/'resume/BPR__s2020.pt').exists()
    assert len(list((output/'per_user_TO').glob('*.csv.gz')))==4
    again=run(0)
    assert '[skip] BPR seed=2020' in again.stdout


@pytest.mark.skipif(os.environ.get('IKGR_TEST_CUDA')!='1',reason='explicit optional CUDA integration probe')
def test_cuda_pause_matches_uninterrupted(tmp_path):
    worker(tmp_path/'baseline','IKGR','--plain','--gpu')
    worker(tmp_path/'paused','IKGR','--gpu','--pause-epoch','1',expected=75)
    worker(tmp_path/'paused','IKGR','--gpu','--resume')
    equal(torch.load(tmp_path/'baseline/final.pt',map_location='cpu'),
          torch.load(tmp_path/'paused/final.pt',map_location='cpu'))
