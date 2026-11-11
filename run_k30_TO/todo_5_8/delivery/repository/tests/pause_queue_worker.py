"""Tiny end-to-end eval_slices queue test, with an injected user pause request."""
import os
from pathlib import Path
import sys

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
os.chdir(root)
import eval_slices as ev
from recbole.trainer import Trainer
from ikgr_core.pause_resume import Control, PauseRequested

original=Trainer._train_epoch
def request_after_epoch(self,data,epoch,*a,**kw):
    result=original(self,data,epoch,*a,**kw)
    if os.environ.get('TEST_PAUSE_AFTER_FIRST')=='1' and epoch==0:
        Control(os.environ['IKGR_CONTROL_DIR']).ask_pause()
    return result
Trainer._train_epoch=request_after_epoch
try:
    ev.main()
except PauseRequested:
    raise SystemExit(75)
