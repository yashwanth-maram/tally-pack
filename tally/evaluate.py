"""Tally Pack - PackBench evaluation: SAM 3.1 tracks -> ledger -> verifier vs ground truth.

Outcome classes (what the system would tell a packer):
  PASS          no violations: the box matches the order and the evidence is consistent
  FAIL          C1 order violation: the ledger proves the box does not match the order
  NEEDS_REVIEW  no order violation, but integrity checks (C2-C4) say the evidence is unreliable
                -> this is what the investigator agent takes over

The metric that matters most is the false-clearance rate: a wrongly packed box marked PASS.
"""
import json
import os

from . import ledger as L

CONTAINER = "cardboard box"


def order_by_prompt(gt):
    """Order expressed per SAM prompt, with 0 for every SKU present on the table but not ordered."""
    return {gt["prompts"][s]: gt["order"].get(s, 0) for s in gt["prompts"]}


def outcome(report):
    checks = {v["check"] for v in report["violations"]}
    if "C1_order" in checks:
        return "FAIL"
    return "NEEDS_REVIEW" if checks else "PASS"


def score_clip(tracks, gt, params=None):
    prompts = sorted(set(gt["prompts"].values()))
    led = L.build_ledger(tracks, CONTAINER, prompts, params)
    rep = L.verify(led, order_by_prompt(gt), params)
    pred = outcome(rep)
    counts = {p: rep["counts"][p]["in_container"] for p in prompts}
    truth = {gt["prompts"][s]: n for s, n in gt["in_box"].items()}
    count_ok = all(counts.get(p, 0) == truth.get(p, 0) for p in prompts)
    return {"name": gt["name"], "scenario": gt["scenario"], "expected": gt["expected"], "pred": pred,
            "count_exact": count_ok, "counts": counts, "truth": truth,
            "violations": [f"{v['check']}:{v['detail']}" for v in rep["violations"]], "model_calls": rep["model_calls"]}


def summarize(results):
    n = len(results)
    fail_gt = [r for r in results if r["expected"] == "FAIL"]
    pass_gt = [r for r in results if r["expected"] == "PASS"]
    false_clear = [r for r in fail_gt if r["pred"] == "PASS"]
    caught = [r for r in fail_gt if r["pred"] == "FAIL"]
    cleared = [r for r in pass_gt if r["pred"] == "PASS"]
    review = [r for r in results if r["pred"] == "NEEDS_REVIEW"]
    return {
        "clips": n,
        "errors_caught": f"{len(caught)}/{len(fail_gt)}",
        "false_clearances": f"{len(false_clear)}/{len(fail_gt)}",
        "correct_clears": f"{len(cleared)}/{len(pass_gt)}",
        "needs_review": f"{len(review)}/{n}",
        "count_exact": f"{sum(r['count_exact'] for r in results)}/{n}",
        "model_calls": sum(r["model_calls"] for r in results),
    }


def run(frames_root, gts, tracks_dir, prompts_for=None, track_fn=None, load_fn=None, save_fn=None):
    """Track (or load cached tracks for) every clip, then score. Returns (results, summary)."""
    os.makedirs(tracks_dir, exist_ok=True)
    results = []
    for gt in gts:
        path = os.path.join(tracks_dir, f"{gt['name']}.json")
        if os.path.exists(path) and load_fn:
            tracks = load_fn(path)
        else:
            prompts = [CONTAINER] + sorted(set(gt["prompts"].values()))
            tracks = track_fn(os.path.join(frames_root, gt["name"]), prompts)
            save_fn(tracks, path)
        results.append(score_clip(tracks, gt))
    return results, summarize(results)


# ---------------------------------------------------------------- v2: category tracking + SKU identity
from . import identity as I  # noqa: E402

CATEGORY_PROMPTS = {"packet": "packet", "can": "can", "bottle": "bottle"}


def category_skus(gt, cat_prompts=None):
    cat_prompts = cat_prompts or CATEGORY_PROMPTS
    out = {}
    for sku, cat in gt["categories"].items():
        out.setdefault(cat_prompts[cat], []).append(sku)
    return out


def oracle_tracks(gt, cat_prompts=None):
    """Perfect perception built from the generator's ground truth (items under a covering hand are unseen)."""
    cat_prompts = cat_prompts or CATEGORY_PROMPTS
    prompts = {p: {} for p in set(cat_prompts[c] for c in gt["categories"].values())}
    for f, boxes in enumerate(gt["truth"]):
        for p in prompts:
            prompts[p][f] = []
        for b in boxes:
            if b["covered"]:
                continue
            p = cat_prompts[gt["categories"][b["sku"]]]
            prompts[p][f].append({"id": b["idx"], "p": 1.0, "box": b["box"], "c": b["c"],
                                  "area": b["box"][2] * b["box"][3]})
    return {"n_frames": gt["n_frames"], "fps": gt["fps"], "size": gt["size"], "prompts": prompts}


def score_clip_v2(tracks, gt, frames_dir, refs, container_tracks=None, oracle_identity=False, params=None):
    cats = category_skus(gt)
    t = dict(tracks)
    t["prompts"] = dict(tracks["prompts"])
    if container_tracks is not None:
        t["prompts"][CONTAINER] = container_tracks
    led = L.build_ledger(t, CONTAINER, list(cats), params)
    integrity = L.verify(led, {}, params)                       # C2-C4 per category
    if oracle_identity:
        idx_sku = {b["idx"]: b["sku"] for b in gt["truth"][0]}
        assignment = {p: {oid: {"sku": idx_sku[oid], "score": 1.0, "runner_up": 0.0, "margin": 1.0}
                          for oid in led["items"][p]} for p in cats}
    else:
        assignment = I.assign(frames_dir, t, cats, refs)
    counts, ambiguous = I.sku_counts(led, assignment)
    order = {s: gt["order"].get(s, 0) for s in gt["categories"]}
    violations = I.check_order(counts, ambiguous, order) + integrity["violations"]
    checks = {v["check"] for v in violations}
    # a verdict is only issued on consistent evidence: any integrity issue (C2-C5) -> NEEDS_REVIEW
    if checks - {"C1_order"}:
        pred = "NEEDS_REVIEW"
    else:
        pred = "FAIL" if "C1_order" in checks else "PASS"
    truth = gt["in_box"]
    count_ok = all(counts.get(s, 0) == truth.get(s, 0) for s in set(truth) | set(counts)) and not ambiguous
    return {"name": gt["name"], "scenario": gt["scenario"], "expected": gt["expected"], "pred": pred,
            "count_exact": count_ok, "counts": counts, "truth": truth, "ambiguous": len(ambiguous),
            "violations": [f"{v['check']}:{v['detail']}" for v in violations], "model_calls": 0}


# ---------------------------------------------------------------- fault injection + verified self-repair
import copy  # noqa: E402


def inject_id_switch(tracks, prompt, oid, at_frame, new_id=None):
    """Simulate a tracker ID switch: from `at_frame` on, object `oid` is reported under a new ID."""
    t = copy.deepcopy(tracks)
    new_id = new_id if new_id is not None else 900 + int(oid)
    for f, dets in t["prompts"][prompt].items():
        if int(f) >= at_frame:
            for d in dets:
                if d["id"] == oid:
                    d["id"] = new_id
    return t, new_id


def propose_repairs(tracks, gt, params=None):
    """Deterministic repair proposals from C4 violations: merge the new ID back into the vanished one."""
    cats = category_skus(gt)
    led = L.build_ledger(tracks, CONTAINER, list(cats), params)
    rep = L.verify(led, {}, params)
    props = []
    for v in rep["violations"]:
        if v["check"] == "C4_identity" and len(v["ids"]) == 2:
            props.append({"type": "merge", "item": v["item"], "keep": v["ids"][1], "drop": v["ids"][0],
                          "why": v["detail"]})
    return props


def self_repair(tracks, gt, frames_dir, refs, max_steps=3, params=None):
    """Apply proposals one at a time through the verifier gate (L.try_repair). Returns (tracks, trace)."""
    cats = category_skus(gt)
    trace = []
    for _ in range(max_steps):
        props = propose_repairs(tracks, gt, params)
        if not props:
            break
        applied = False
        for a in props:
            r = L.try_repair(tracks, CONTAINER, list(cats), {}, a, params)
            trace.append({"action": a, "accepted": r["accepted"],
                          "violations_before": len(r["before"]["violations"]),
                          "violations_after": len(r["after"]["violations"])})
            if r["accepted"]:
                tracks, applied = r["tracks"], True
                break
        if not applied:
            break
    return tracks, trace
