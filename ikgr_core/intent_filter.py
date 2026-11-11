"""Deterministic removal of intents with the most distinct TRAIN item links."""
import math
import numpy as np
import torch


def filter_intents(pack, train_item_tokens, top_pct=0):
    pct = float(top_pct)
    if not math.isfinite(pct) or not 0 <= pct <= 100:
        raise ValueError("intent_filter_top_pct must be between 0 and 100")
    count = int(pack["n_intents"])
    if pack["intent_emb"].shape[0] != count:
        raise ValueError("intent embedding rows do not match n_intents")
    train_tokens = {str(t) for t in train_item_tokens}
    degree = np.zeros(count, dtype=np.int64)
    for token, ids in pack["item_intents"].items():
        if str(token) not in train_tokens:
            continue
        unique = np.asarray(sorted(set(int(i) for i in ids)), dtype=np.int64)
        if len(unique) and (unique.min() < 0 or unique.max() >= count):
            raise ValueError("intent ID outside embedding range")
        degree[unique] += 1
    active = np.flatnonzero(degree)
    ordered = sorted(active.tolist(), key=lambda i: (-int(degree[i]), i))
    removed = ordered[:math.floor(len(active) * pct / 100)]
    blocked = set(removed)
    stats = {"top_pct": pct, "n_intents": count, "active_train_item_intents": len(active),
             "train_items": len(train_tokens), "removed_count": len(removed),
             "removed": [{"intent_id": i, "train_item_degree": int(degree[i])} for i in removed],
             "definition": "distinct train items; positive-degree intents only; floor; degree DESC, ID ASC; both edge types"}
    output = pack if not removed else dict(pack)
    for key in ("user_intents", "item_intents"):
        before = pack[key]
        if removed:
            after = {token: torch.tensor([int(i) for i in ids if int(i) not in blocked], dtype=torch.long)
                     for token, ids in before.items()}
            output[key] = after
        else:
            after = before
        stats[key] = {"pack_edges_before": sum(len(v) for v in before.values()),
                      "pack_edges_after": sum(len(v) for v in after.values()),
                      "pack_empty_before": sum(len(v) == 0 for v in before.values()),
                      "pack_empty_after": sum(len(v) == 0 for v in after.values())}
    return output, stats
