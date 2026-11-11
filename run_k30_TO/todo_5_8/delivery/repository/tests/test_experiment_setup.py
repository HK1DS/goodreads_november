from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import eval_slices as ev
from ikgr_core.intent_filter import filter_intents


def pack():
    return {"n_intents": 5, "intent_emb": torch.arange(20).reshape(5,4).float(),
            "item_intents": {"a":torch.tensor([0,0,1]), "b":torch.tensor([0,2]),
                             "c":torch.tensor([1,2]), "heldout":torch.tensor([3])},
            "user_intents": {"u":torch.tensor([0,1]), "v":torch.tensor([0]), "w":torch.tensor([4])}}


def test_filter_train_degree_ties_nested_and_both_edges():
    p = pack()
    sets=[]
    for pct,expected in [(0,[]),(34,[0]),(67,[0,1]),(100,[0,1,2])]:
        out,stats = filter_intents(p,{"a","b","c"},pct)
        removed = [r["intent_id"] for r in stats["removed"]]
        assert removed == expected
        assert stats["active_train_item_intents"] == 3  # held-out and user-only intents excluded
        assert all(r["train_item_degree"] == 2 for r in stats["removed"])  # duplicate edge counted once
        assert out["intent_emb"] is p["intent_emb"] and out["n_intents"] == 5
        for key in ("user_intents","item_intents"):
            assert all(not set(ids.tolist()).intersection(removed) for ids in out[key].values())
        sets.append(set(removed))
    assert all(a<=b for a,b in zip(sets,sets[1:]))
    assert p["item_intents"]["a"].tolist() == [0,0,1]  # original is immutable
    out,_=filter_intents(p,{"a","b","c"},34)
    assert out["item_intents"]["a"].tolist()==[1]
    assert out["user_intents"]["v"].numel()==0


def test_zero_filter_and_no_active_intents_are_noops():
    p=pack()
    assert filter_intents(p,{"a","b"},0)[0] is p
    assert filter_intents(p,set(),20)[0] is p


@pytest.mark.parametrize("pct",[-1,101,float("nan"),float("inf")])
def test_bad_filter_percentage_rejected(pct):
    with pytest.raises(ValueError): filter_intents(pack(),{"a"},pct)


def test_eval_batch_converts_users_to_score_count_and_requires_cuda():
    config={"eval_users_per_batch":128,"eval_batch_size":4096,"require_cuda":True,"device":torch.device("cuda")}
    ev._configure_eval_batch(config,SimpleNamespace(item_num=27976))
    assert config["eval_batch_size"]==3580928
    config["device"]=torch.device("cpu")
    with pytest.raises(RuntimeError): ev._configure_eval_batch(config,SimpleNamespace(item_num=10))


@pytest.mark.parametrize("users",[0,-1,1.5,True])
def test_bad_eval_batch_rejected(users):
    with pytest.raises(ValueError):
        ev._configure_eval_batch({"eval_users_per_batch":users},SimpleNamespace(item_num=10))


def test_actual_interaction_change_invalidates_result(tmp_path):
    atomic=tmp_path/"ds"/"ds.inter"
    atomic.parent.mkdir()
    atomic.write_text("before")
    paths={"inter_file":str(tmp_path/"interactions.csv")}
    a,_=ev._experiment_context("BPR","BPR",{},{"dataset":"ds"},paths,"TO")
    atomic.write_text("changed interaction data")
    b,_=ev._experiment_context("BPR","BPR",{},{"dataset":"ds"},paths,"TO")
    assert a != b


def test_runtime_fingerprint_handles_hidden_gpu(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    def invalid_query(*args):
        raise AssertionError("No visible GPU may be queried")
    monkeypatch.setattr(torch.cuda, "get_device_name", invalid_query)
    _,ctx=ev._experiment_context("BPR","BPR",{},{"dataset":"ds"},{"inter_file":str(tmp_path/"input.csv")},"TO")
    assert ctx["runtime"]["gpu"] is None


def test_resume_requires_requested_per_user_files(tmp_path):
    results={"BPR":{"seeds":{"2020":{}}}}
    assert not ev._seed_complete(results,["BPR"],2020,tmp_path,False)
    (tmp_path/"BPR__s2020__test.csv.gz").write_bytes(b"fixture")
    assert ev._seed_complete(results,["BPR"],2020,tmp_path,False)
    assert not ev._seed_complete(results,["BPR"],2020,tmp_path,True)
    (tmp_path/"BPR__s2020__valid.csv.gz").write_bytes(b"fixture")
    assert ev._seed_complete(results,["BPR"],2020,tmp_path,True)
    artifact = tmp_path.parent / (tmp_path.name + "_filter.json")
    artifact.write_text("{}")
    results["BPR"]["seeds"]["2020"]["intent_filter"] = {"file":artifact.name, "sha256":ev._file_sha256(artifact)}
    assert ev._seed_complete(results,["BPR"],2020,tmp_path,True)
    artifact.write_text("changed")
    assert not ev._seed_complete(results,["BPR"],2020,tmp_path,True)
    artifact.unlink()
    assert not ev._seed_complete(results,["BPR"],2020,tmp_path,True)
