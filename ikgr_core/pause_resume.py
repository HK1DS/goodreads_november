"""Request-only, epoch-boundary pause/resume for the fixed RecBole 1.2 protocol."""
import json
import os
from pathlib import Path
import random
import time

PAUSED_EXIT = 75


class PauseRequested(Exception):
    pass


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


class RunLock:
    """OS lock, released on process exit; the persistent file is not the lock."""
    def __init__(self, folder):
        self.path = Path(folder) / 'runner.lock'

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open('a+b')
        self.file.seek(0, 2)
        if self.file.tell() == 0:
            self.file.write(b'0')
            self.file.flush()
        self.file.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.file.close()
            raise RuntimeError('An experiment launcher is already running for this output directory') from exc
        return self

    def __exit__(self, *args):
        self.file.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(self.file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(self.file.fileno(), fcntl.LOCK_UN)
        self.file.close()


class Control:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.request = self.folder / 'pause.request'
        self.status_file = self.folder / 'status.json'

    def requested(self):
        return self.request.is_file()

    def status(self, state, **details):
        atomic_json(self.status_file, dict(state=state, time=time.time(), pid=os.getpid(), **details))

    def event(self, event, **details):
        self.folder.mkdir(parents=True, exist_ok=True)
        with (self.folder/'events.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(dict(event=event,time=time.time(),pid=os.getpid(),**details))+'\n')
            f.flush()
            os.fsync(f.fileno())

    def read(self):
        value = json.loads(self.status_file.read_text(encoding='utf-8')) if self.status_file.exists() else {'state':'idle'}
        value['pause_requested'] = self.requested()
        return value

    def ask_pause(self):
        self.folder.mkdir(parents=True, exist_ok=True)
        self.request.touch(exist_ok=True)
        return self.read()

    def boundary(self):
        if self.requested():
            raise PauseRequested('Pause requested between runs; completed results are preserved')


def _atomic_save(path, value):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('wb') as f:
        torch.save(value, f, pickle_protocol=4)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def _rng(loaders):
    import numpy as np
    import torch
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'loaders': [loader.generator.get_state() for loader in loaders]}


def _restore_rng(value, loaders):
    import numpy as np
    import torch
    random.setstate(value['python'])
    np.random.set_state(value['numpy'])
    torch.set_rng_state(value['torch'].cpu())
    if value['cuda']:
        torch.cuda.set_rng_state_all([s.cpu() for s in value['cuda']])
    for loader, state in zip(loaders, value['loaders']):
        loader.generator.set_state(state.cpu())


def fit_with_pause(trainer, train, valid, test, fit_info, control, checkpoint, identity, allow_resume=False):
    """Keep RecBole.fit unchanged; inspect requests before each NEXT epoch and after fit.

    The next-epoch boundary is after previous validation/early-stopping/checkpoint updates.
    Workers/AMP/distributed training are excluded because their extra state is not captured.
    """
    import torch
    import recbole
    if recbole.__version__ != '1.2.0':
        raise RuntimeError('Pause/resume was validated with RecBole 1.2.0 only')
    if (train.num_workers != 0 or not trainer.config['single_spec'] or trainer.enable_amp or trainer.enable_scaler
            or trainer.config['train_neg_sample_args']['distribution'] != 'uniform'):
        raise RuntimeError('Pause/resume requires worker=0, single-process, uniform negatives, full precision')
    checkpoint = Path(checkpoint)
    loaders = [train, valid, test]
    last_epoch = -1
    elapsed_before = 0.0
    complete = False
    if checkpoint.exists():
        if not allow_resume:
            raise RuntimeError('A paused checkpoint exists; use --resume')
        saved = torch.load(checkpoint, map_location='cpu')
        if saved.get('version') != 1 or saved['identity'] != identity:
            raise RuntimeError('Paused checkpoint code/data/config/seed identity differs; refusing unsafe resume')
        trainer.model.load_state_dict(saved['model'])
        trainer.model.load_other_parameter(saved['other'])
        trainer.optimizer.load_state_dict(saved['optimizer'])
        for key, value in saved['trainer'].items():
            setattr(trainer, key, value)
        last_epoch = saved['last_epoch']
        trainer.start_epoch = last_epoch + 1
        fit_info.update(saved['fit_info'])
        elapsed_before = saved['train_seconds']
        complete = saved['fit_complete']
        if saved['best_model'] is not None:
            _atomic_save(trainer.saved_model_file, saved['best_model'])
        _restore_rng(saved['rng'], loaders)
        del saved
        control.event('resumed', **identity, next_epoch=last_epoch+1, fit_complete=complete)
        print(f'[resume] {identity["spec"]} seed={identity["seed"]} next_epoch={last_epoch + 1}', flush=True)
    original = trainer._train_epoch
    started = time.perf_counter()

    def pause_if_requested(done):
        if not control.requested():
            return
        rng = _rng(loaders)
        best = None
        if fit_info.get('best_epoch') is not None:
            full = torch.load(trainer.saved_model_file, map_location='cpu')
            best = {key:full[key] for key in ('state_dict','other_parameter','epoch')}
        payload = {'version':1, 'identity':identity, 'model':trainer.model.state_dict(),
                   'other':trainer.model.other_parameter(), 'optimizer':trainer.optimizer.state_dict(),
                   'trainer':{key:getattr(trainer,key) for key in
                              ('cur_step','best_valid_score','best_valid_result','train_loss_dict')},
                   'last_epoch':last_epoch, 'fit_complete':done, 'fit_info':dict(fit_info),
                   'train_seconds':elapsed_before + time.perf_counter()-started,
                   'rng':rng, 'best_model':best}
        start_save = time.perf_counter()
        _atomic_save(checkpoint, payload)
        control.status('checkpoint_saved', **identity, last_epoch=last_epoch, fit_complete=done,
                       checkpoint=str(checkpoint.resolve()), save_seconds=time.perf_counter()-start_save)
        control.event('checkpoint_saved', **identity, last_epoch=last_epoch, fit_complete=done,
                      checkpoint=str(checkpoint.resolve()), save_seconds=time.perf_counter()-start_save)
        print(f'[pause] saved after epoch={last_epoch}; exiting to release GPU',flush=True)
        raise PauseRequested(str(checkpoint))

    def train_epoch(data, epoch, *args, **kwargs):
        nonlocal last_epoch
        pause_if_requested(False)
        control.status('training', **identity, epoch=epoch, max_epochs=trainer.epochs)
        result = original(data, epoch, *args, **kwargs)
        last_epoch = epoch
        fit_info['last_epoch'] = epoch
        return result

    trainer._train_epoch = train_epoch
    try:
        # RecBole.fit would overwrite the best checkpoint when start_epoch >= epochs.
        result = (trainer.best_valid_score, trainer.best_valid_result) if complete else trainer.fit(
            train, valid, saved=True, show_progress=False)
        pause_if_requested(True)
        return result, elapsed_before + time.perf_counter()-started
    finally:
        trainer._train_epoch = original

