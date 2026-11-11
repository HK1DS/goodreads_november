"""Separate-process synthetic training for pause/resume equivalence tests."""
import argparse
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
ap=argparse.ArgumentParser()
ap.add_argument('--folder',required=True)
ap.add_argument('--model',default='BPR')
ap.add_argument('--pause-epoch',type=int,default=-1)
ap.add_argument('--resume',action='store_true')
ap.add_argument('--plain',action='store_true')
ap.add_argument('--gpu',action='store_true')
ap.add_argument('--signature',default='fixture-v1')
args=ap.parse_args()
sys.argv=[sys.argv[0]]
os.environ['CUDA_VISIBLE_DEVICES']='0' if args.gpu else ''
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import numpy as np
import torch
import eval_slices as ev
from ikgr_core.pause_resume import Control, PauseRequested, fit_with_pause

folder=Path(args.folder)
folder.mkdir(parents=True,exist_ok=True)
dsdir=folder/'tiny'
dsdir.mkdir(exist_ok=True)
inter=dsdir/'tiny.inter'
if not inter.exists():
    rng=np.random.default_rng(19)
    lines=['user_id:token\titem_id:token\ttimestamp:float']
    for u in range(1,25):
        for t,i in enumerate(rng.choice(np.arange(1,51),24,replace=False)):
            lines.append(f'{u}\t{i}\t{t+1}')
    inter.write_text('\n'.join(lines)+'\n')
kg=folder/'kg.pt'
if not kg.exists():
    torch.save({'n_intents':8,'intent_emb':torch.arange(128).reshape(8,16).float()/128,
                'user_intents':{str(u):torch.tensor([u%8,(u+1)%8]) for u in range(1,25)},
                'item_intents':{str(i):torch.tensor([i%8,(i+1)%8]) for i in range(1,51)}},kg)
ev._set_determinism(2020)
model_arg='BPR' if args.model=='BPR' else ev.IKGRModel
cd={'data_path':str(folder),'USER_ID_FIELD':'user_id','ITEM_ID_FIELD':'item_id','TIME_FIELD':'timestamp',
    'load_col':{'inter':['user_id','item_id','timestamp']},'seed':2020,'reproducibility':True,
    'embedding_size':16,'train_batch_size':64,'eval_batch_size':1024,'epochs':8,'eval_step':2,
    'stopping_step':1,'valid_metric':'NDCG@5','metrics':['NDCG','Recall'],'topk':[5],
    'eval_args':{'split':{'RS':[.8,.1,.1]},'order':'TO','group_by':'user','mode':'full'},
    'train_neg_sample_args':{'distribution':'uniform'},'worker':0,'gpu_id':'0' if args.gpu else '',
    'checkpoint_dir':str(folder/'best'),'show_progress':False,
    'use_kg':True,'kg_pack_path':str(kg),'kg_layers':1,'kg_cap':4,'intent_learnable':False,
    'use_dynamic':True,'intent_filter_top_pct':20,'use_meta':False}
cfg=ev.Config(model=model_arg,dataset='tiny',config_dict=cd)
ds=ev.create_dataset(cfg)
train,valid,test=ev.data_preparation(cfg,ds)
klass=ev.get_model('BPR') if args.model=='BPR' else model_arg
model=klass(cfg,train.dataset).to(cfg['device'])
if getattr(model,'use_dynamic',False): ev._build_recency(model,train,cfg)
trainer=ev.get_trainer(cfg['MODEL_TYPE'],cfg['model'])(cfg,model)
trainer.saved_model_file=str(folder/'best/model.pth')
fit_info={'best_epoch':None,'last_epoch':None}
save=trainer._save_checkpoint
def tracked_save(epoch,*a,**kw):
    fit_info['best_epoch']=epoch
    return save(epoch,*a,**kw)
trainer._save_checkpoint=tracked_save
# Real evaluation runs; deliberately lower later selection scores force stopping at epoch 5.
evaluate=trainer._valid_epoch
def fixed_selection(*a,**kw):
    _,metrics=evaluate(*a,**kw)
    return (1. if fit_info['last_epoch']==1 else .5),metrics
trainer._valid_epoch=fixed_selection
control=Control(folder/'control')
if args.resume: control.request.unlink(missing_ok=True)
epoch_step=trainer._train_epoch
def request_after_epoch(data,epoch,*a,**kw):
    loss=epoch_step(data,epoch,*a,**kw)
    fit_info['last_epoch']=epoch
    if epoch==args.pause_epoch: control.ask_pause()
    return loss
trainer._train_epoch=request_after_epoch
try:
    if args.plain:
        result=trainer.fit(train,valid,saved=True,show_progress=False)
    else:
        result,_=fit_with_pause(trainer,train,valid,test,fit_info,control,folder/'resume.pt',
                                {'spec':args.model,'seed':2020,'signature':args.signature},args.resume)
except PauseRequested:
    trainer.tensorboard.close()
    raise SystemExit(75)
trainer.tensorboard.close()
final={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
best=torch.load(trainer.saved_model_file,map_location='cpu')
model.load_state_dict(best['state_dict'])
model.eval()
if hasattr(model,'eval_cache_items'): model.eval_cache_items()
with torch.no_grad():
    interaction=next(iter(test))[0].to(cfg['device'])
    scores=model.full_sort_predict(interaction).cpu()
torch.save({'final':final,'optimizer':trainer.optimizer.state_dict(),'best':best['state_dict'],
            'best_epoch':best['epoch'],'fit_info':fit_info,'best_score':trainer.best_valid_score,
            'cur_step':trainer.cur_step,'losses':trainer.train_loss_dict,'scores':scores,
            'next_train_batch':next(iter(train)).interaction,'python_next':random.random(),
            'numpy_next':np.random.random(8),'torch_next':torch.rand(8),
            'cuda_next':torch.rand(8,device='cuda').cpu() if args.gpu else None},folder/'final.pt')
print('WORKER PASSED',flush=True)
