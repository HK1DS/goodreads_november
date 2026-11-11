"""Prepare/check the experiment; training requires explicit --run or --resume."""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PHASES = {"5": ["BPR", "IKGR_kgoff", "IKGR_dyn_nointent"],
          "8": ["IKGR_dyn", "IKGR_dyn_filter5", "IKGR_dyn_filter10", "IKGR_dyn_filter20"]}


def experiment_environment(config, specs):
    env = dict(os.environ)
    for key in list(env):
        if key.startswith("IKGR_"):
            del env[key]
    env.update(IKGR_CONFIG=str(config), IKGR_SPLIT="TO", IKGR_SEEDS="2020,2021,2022",
               IKGR_SPECS=",".join(specs), IKGR_EPOCHS="40", IKGR_EVAL_STEP="2",
               IKGR_STOPPING_STEP="3", IKGR_VALID_METRIC="NDCG@10", IKGR_LR="0.001", IKGR_REG="0.000001",
               IKGR_POP_RERANK_SPECS=",".join(specs), IKGR_POP_RERANK_GRID="0.25,0.5,1.0",
               IKGR_DUMP_VALID="1", IKGR_REQUIRE_PROVENANCE="1", PYTHONIOENCODING="utf-8", PYTHONUTF8="1",
               PYTHONHASHSEED="2020", CUBLAS_WORKSPACE_CONFIG=":4096:8",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", WANDB_MODE="disabled")
    return env


def check(config_path, specs, output):
    import numpy as np
    import torch
    import yaml
    import eval_slices as ev
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    paths, rb = cfg["paths"], cfg["recbole"]
    assert rb["embedding_size"] == 64 and rb["epochs"] == 40
    assert rb["train_batch_size"] == 2048 and rb["eval_users_per_batch"] == 128
    assert rb["eval_step"] == 2 and rb["stopping_step"] == 3
    assert rb["require_cuda"] and torch.cuda.is_available(), "CUDA is required"
    ev._set_determinism(2020)
    config = ev.Config(model="BPR", dataset=rb["dataset"], config_dict=ev._config(rb, paths, {}, 2020))
    dataset = ev.create_dataset(config)
    ev._configure_eval_batch(config, dataset)
    train, valid, test = ev.data_preparation(config, dataset)
    assert valid.step == test.step == 128
    assert len(valid.user_df) == len(test.user_df) == 62142, "Full evaluation population changed"
    guard = ev._check_profile_leakage(paths, {"use_kg": True, "kg_pack_path": paths["kg_pack"]}, train, config, "TO")
    assert guard["verified"]
    batch = next(iter(train)).to(config["device"])
    eval_batch = next(iter(valid))[0].to(config["device"])
    manifest = {"purpose": "preflight only; no optimizer steps, epochs or formal results",
                "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda,
                "torch_threads": torch.get_num_threads(), "train_interactions": len(train.dataset.inter_feat),
                "valid_users": len(valid.user_df), "test_users": len(test.user_df),
                "eval_users_per_batch": valid.step, "eval_batch_size": config["eval_batch_size"],
                "seeds": [2020,2021,2022], "specs": specs, "leakage_guard": guard, "checks": {}}
    output.mkdir(parents=True, exist_ok=True)
    all_specs = ev.build_specs(paths)
    removal_sets = {}
    for name in specs:
        ev._set_determinism(2020)
        model_arg, extra = all_specs[name]
        model_cfg = ev.Config(model=model_arg, dataset=rb["dataset"], config_dict=ev._config(rb, paths, extra, 2020))
        ev._configure_eval_batch(model_cfg, dataset)
        klass = model_arg if not isinstance(model_arg,str) else ev.get_model(model_cfg["model"])
        model = klass(model_cfg, train.dataset).to(model_cfg["device"])
        if getattr(model,"use_dynamic",False):
            ev._build_recency(model, train, model_cfg)
        model.train()
        loss = model.calculate_loss(batch)
        loss = sum(loss) if isinstance(loss,tuple) else loss
        assert torch.isfinite(loss).item(), f"{name}: non-finite loss"
        loss.backward()
        assert all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None)
        model.eval()
        with torch.no_grad():
            scores = model.full_sort_predict(eval_batch).reshape(len(eval_batch), -1)
            assert scores.shape == (128, dataset.item_num)
            assert torch.isfinite(scores).all().item()
        signature, context = ev._experiment_context(name, model_arg, extra, rb, paths, "TO")
        entry = {"cuda": next(model.parameters()).device.type == "cuda", "finite_loss_and_gradients": True,
                 "finite_full_sort_scores": True, "signature": signature, "context": context}
        stats = getattr(model,"intent_filter_stats",None)
        if stats is not None:
            pct = stats["top_pct"]
            removed = {r["intent_id"] for r in stats["removed"]}
            removal_sets[pct] = removed
            for prefix in ("user", "item"):
                ids = getattr(model,prefix+"_intent_ids")
                mask = getattr(model,prefix+"_intent_mask")
                actual = set(ids[mask].cpu().numpy().tolist())
                assert not removed.intersection(actual), f"{name}: removed intent remains in effective edges"
            ev._atomic_json(output/f"{name}_filter.json", stats)
            entry["filter"] = {"pct": pct, "removed_count": len(removed), "active": stats["active_train_item_intents"]}
        manifest["checks"][name] = entry
        print(f"PREFLIGHT {name}: GPU forward/backward/full-sort OK" + (f"; removed={len(removed)}" if stats else ""),flush=True)
        del scores, loss, model
        gc.collect()
        torch.cuda.empty_cache()
    sorted_pcts = sorted(removal_sets)
    assert all(removal_sets[a] <= removal_sets[b] for a,b in zip(sorted_pcts,sorted_pcts[1:]))
    manifest["filter_sets_nested"] = True
    manifest["status"] = "passed"
    ev._atomic_json(output/"preflight.json", manifest)
    print(f"PREFLIGHT PASSED -> {output/'preflight.json'}",flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=["5","8","all"], default="all")
    ap.add_argument("--config", default="configs/todo_5_8.yaml")
    actions = ap.add_mutually_exclusive_group()
    actions.add_argument("--check", action="store_true", help="default: validate full-data settings and GPU paths")
    actions.add_argument("--run", action="store_true", help="explicitly start the approved formal training jobs")
    actions.add_argument("--pause", action="store_true", help="request checkpoint + exit at the next safe epoch boundary")
    actions.add_argument("--resume", action="store_true", help="clear pause request and continue saved state")
    actions.add_argument("--status", action="store_true", help="read status without loading CUDA")
    ap.add_argument("--check-output", default="../settings_preflight")
    args = ap.parse_args()
    os.chdir(ROOT)
    sys.path.insert(0,str(ROOT))
    config = Path(args.config).resolve()
    import yaml
    from ikgr_core.pause_resume import Control, RunLock, PAUSED_EXIT
    cfg = yaml.safe_load(config.read_text(encoding='utf-8'))
    control = Control(Path(cfg['paths']['workdir']).resolve() / 'control')
    if args.pause or args.status:
        print(json.dumps(control.ask_pause() if args.pause else control.read(), ensure_ascii=False, indent=2))
        return
    specs = PHASES["5"] + PHASES["8"] if args.phase == "all" else PHASES[args.phase]
    env = experiment_environment(config, specs)
    env['CUDA_VISIBLE_DEVICES'] = str(cfg['recbole'].get('gpu_id', '0'))
    env['IKGR_CONTROL_DIR'] = str(control.folder)
    env['IKGR_RESUME_ALLOWED'] = '1' if args.resume else '0'
    # Remove inherited IKGR knobs in this process as well as in child processes.
    for key in list(os.environ):
        if key.startswith("IKGR_"):
            del os.environ[key]
    os.environ.update(env)
    sys.argv = [sys.argv[0]]  # Do not let RecBole reinterpret launcher flags.
    if not (args.run or args.resume):
        check(config, specs, Path(args.check_output).resolve())
        return
    with RunLock(control.folder):
        if args.resume:
            previous = control.read()
            if previous.get('phase') and previous['phase'] != args.phase:
                raise RuntimeError('Resume with the same --phase as the original launch')
            control.request.unlink(missing_ok=True)
        elif control.requested() or list((control.folder.parent/'resume').glob('*.pt')):
            raise RuntimeError('Paused state exists; use --resume')
        control.status('preparing', phase=args.phase, gpu_released=False)
        try:
            # Run preflight in a separate process so it releases its CUDA context before training.
            # A resume reconstructs and fingerprints its own model/data; no redundant 7-model preflight.
            if args.run:
                subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), '--check', '--phase', args.phase,
                                '--config', str(config), '--check-output', args.check_output], cwd=ROOT, env=env, check=True)
            if control.requested():
                code = PAUSED_EXIT
            else:
                code = subprocess.run([sys.executable,'-u','eval_slices.py'],cwd=ROOT,env=env).returncode
            details = control.read()
            for key in ('state','time','pid','pause_requested','phase','gpu_released'):
                details.pop(key,None)
            control.status('paused' if code == PAUSED_EXIT else 'completed' if code == 0 else 'failed',
                           phase=args.phase, gpu_released=True, exit_code=code, **details)
            if code not in (0, PAUSED_EXIT):
                raise RuntimeError(f'Experiment failed with exit code {code}')
            print('PAUSED; child process exited and GPU resources released' if code == PAUSED_EXIT else 'COMPLETED',flush=True)
        except BaseException:
            # Do not claim GPU release if an unexpected parent interruption occurred.
            if control.read().get('state') != 'failed':
                control.status('launcher_error', phase=args.phase)
            raise


if __name__ == "__main__":
    main()
