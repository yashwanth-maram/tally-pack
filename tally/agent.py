"""Tally Pack - investigator agent (Muse Glimmer) behind a deterministic verifier gate.

The verifier issues a verdict only on consistent evidence. When it cannot (NEEDS_REVIEW), the
investigator takes over:

  1. deterministic code turns each violation into concrete candidate repairs
       C4 suspected ID switch      -> merge(new id -> vanished id)
       C3 jump to another object   -> drop the spurious detections between the jump out and back
       C5 ambiguous identity       -> set_identity(id, sku) for each candidate SKU
  2. Muse Glimmer sees the violation, the ledger events and crops of the items involved, and
     picks one candidate or escalates
  3. the repair is applied to a copy and kept only if the verifier confirms it removed
     violations without creating new ones (identity choices need confidence >= 0.8)
  4. every run ends in a stop state: answered | awaiting_signoff | needs_input | paused

Glimmer can never clear a box by itself: only the verifier issues PASS / FAIL.
"""
import copy

import cv2
import numpy as np
from PIL import Image

from . import identity as I
from . import ledger as L

CONTAINER = "cardboard box"
SYSTEM = ("You are the investigator for a warehouse packing verifier. A deterministic verifier found "
          "inconsistent evidence in a camera track of a packing desk. You receive the violation, recent "
          "ledger events, crops of the items involved, and numbered candidate repairs. Pick the repair that "
          "best explains the evidence, or escalate to a human if the crops do not support any candidate. "
          "Think briefly. Answer only JSON: {\"choice\": <candidate number or \"escalate\">, "
          "\"confidence\": <0-1>, \"reason\": \"<one sentence>\"}")


# ------------------------------------------------------------------ audit (no model calls)
def audit(tracks, order, cat_skus, frames_dir, refs, identity_overrides=None, params=None):
    """Ledger + identity + checks. Returns a dict with verdict PASS | FAIL | NEEDS_REVIEW."""
    cat_skus = {p: s for p, s in cat_skus.items() if p in tracks["prompts"]}
    led = L.build_ledger(tracks, CONTAINER, list(cat_skus), params)
    integrity = L.verify(led, {}, params)
    assignment = I.assign(frames_dir, tracks, cat_skus, refs)
    for (prompt, oid), sku in (identity_overrides or {}).items():
        if oid in assignment.get(prompt, {}):
            assignment[prompt][oid] = {"sku": sku, "score": 1.0, "runner_up": 0.0, "margin": 1.0, "override": True}
    counts, ambiguous = I.sku_counts(led, assignment)
    violations = I.check_order(counts, ambiguous, order) + integrity["violations"]
    checks = {v["check"] for v in violations}
    verdict = "NEEDS_REVIEW" if checks - {"C1_order"} else ("FAIL" if "C1_order" in checks else "PASS")
    return {"verdict": verdict, "violations": violations, "counts": counts, "ledger": led,
            "assignment": assignment, "events": L.events_table(led)}


# ------------------------------------------------------------------ candidate repairs
def candidates(rep):
    out, c3 = [], {}
    for v in rep["violations"]:
        if v["check"] == "C4_identity" and len(v["ids"]) == 2:
            out.append({"type": "merge", "item": v["item"], "keep": v["ids"][1], "drop": v["ids"][0], "from": v})
        elif v["check"] == "C3_continuity":
            c3.setdefault((v["item"], v["ids"][0]), []).append(v["frames"])
        elif v["check"] == "C5_identity_ambiguous":
            for sku in v["detail"].split("could be ")[1].split(" (")[0].split("|"):
                out.append({"type": "set_identity", "item": v["item"], "id": v["ids"][0], "sku": sku, "from": v})
    for (item, oid), spans in c3.items():
        spans = sorted(spans)
        for (a1, a2), (b1, b2) in zip(spans, spans[1:]):       # jump out ... jump back
            out.append({"type": "drop_range", "item": item, "id": oid, "f0": a2, "f1": b1,
                        "from": {"check": "C3_continuity", "detail": f"ID {oid} left its path f{a2}-f{b1}"}})
        if len(spans) == 1:
            out.append({"type": "split", "item": item, "id": oid, "at": spans[0][1],
                        "from": {"check": "C3_continuity", "detail": f"ID {oid} jumped at f{spans[0][1]}"}})
    return out


def apply(tracks, a):
    t = copy.deepcopy(tracks)
    rec = t["prompts"][a["item"]]
    if a["type"] == "merge":
        return L.merge_ids(tracks, a["item"], a["keep"], a["drop"])
    if a["type"] == "drop_range":
        for f in range(a["f0"], a["f1"] + 1):
            if f in rec:
                rec[f] = [d for d in rec[f] if d["id"] != a["id"]]
    elif a["type"] == "split":
        new = 800 + int(a["id"])
        for f, dets in rec.items():
            if f >= a["at"]:
                for d in dets:
                    if d["id"] == a["id"]:
                        d["id"] = new
    return t


# ------------------------------------------------------------------ evidence images for Glimmer
def _crop(frames_dir, f, box, label, size=160):
    img = cv2.imread(f"{frames_dir}/{f:05d}.jpg")
    if img is None:
        return np.zeros((size, size, 3), np.uint8)
    x, y, w, h = [int(v) for v in box]
    pad = int(0.4 * max(w, h))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    c = img[y0:y + h + pad, x0:x + w + pad]
    c = cv2.resize(c, (size, size)) if c.size else np.zeros((size, size, 3), np.uint8)
    cv2.putText(c, label, (4, 16), 0, 0.45, (0, 255, 255), 1)
    return c


def evidence_image(tracks, a, frames_dir):
    """Side-by-side crops of the IDs involved, around the frames the candidate is about."""
    rec = tracks["prompts"][a["item"]]
    ids = [a.get("keep"), a.get("drop"), a.get("id")]
    tiles = []
    for oid in [i for i in ids if i is not None]:
        fs = sorted(f for f, ds in rec.items() if any(d["id"] == oid for d in ds))
        for f in ([fs[0], fs[len(fs) // 2], fs[-1]] if fs else []):
            d = next(d for d in rec[f] if d["id"] == oid)
            tiles.append(_crop(frames_dir, f, d["box"], f"id{oid} f{f}"))
    if not tiles:
        tiles = [np.zeros((160, 160, 3), np.uint8)]
    return Image.fromarray(cv2.cvtColor(np.hstack(tiles[:6]), cv2.COLOR_BGR2RGB))


def _describe(c, i):
    if c["type"] == "merge":
        return f"{i}: merge ID {c['drop']} into ID {c['keep']} (same physical item, tracker switched IDs)"
    if c["type"] == "drop_range":
        return f"{i}: drop ID {c['id']} detections f{c['f0']}-f{c['f1']} (tracker briefly latched onto another object)"
    if c["type"] == "split":
        return f"{i}: split ID {c['id']} at f{c['at']} into a new item (two different objects)"
    return f"{i}: item ID {c['id']} is SKU {c['sku']}"


# ------------------------------------------------------------------ the loop
def investigate(tracks, order, cat_skus, frames_dir, refs, ask=None, max_steps=4, params=None):
    """Run the investigator until the verifier can issue a verdict. `ask` = llm.ask_json (None = rules only)."""
    overrides, trace = {}, []
    rep = audit(tracks, order, cat_skus, frames_dir, refs, overrides, params)
    for step in range(max_steps):
        if rep["verdict"] in ("PASS", "FAIL"):
            return {"stop": "answered", "verdict": rep["verdict"], "report": rep, "tracks": tracks, "trace": trace}
        cands = candidates(rep)
        if not cands:
            return {"stop": "awaiting_signoff", "verdict": "NEEDS_REVIEW", "report": rep, "tracks": tracks, "trace": trace,
                    "reason": "no repair can explain: " + "; ".join(v["detail"] for v in rep["violations"][:3])}
        choice, conf, reason, secs = 0, None, "rules: first candidate", 0.0
        if ask is not None:
            viol = "; ".join(f"{v['check']}: {v['detail']}" for v in rep["violations"][:4])
            ev = "; ".join(f"t={e['t']}s id{e['id']} {e['event']}" for e in rep["events"][-8:])
            prompt = (f"Violations: {viol}\nRecent ledger events: {ev}\nCandidate repairs:\n" +
                      "\n".join(_describe(c, i) for i, c in enumerate(cands)) +
                      "\nThe image shows crops of the item IDs involved (first, middle, last sighting).")
            ans, raw, secs = ask(SYSTEM, prompt, [evidence_image(tracks, cands[0], frames_dir)])
            if ans is None:
                reason = "model gave no decision -> rules fallback"
            elif ans.get("choice") == "escalate":
                trace.append({"step": step, "candidates": [_describe(c, i) for i, c in enumerate(cands)],
                              "choice": "escalate", "reason": ans.get("reason"), "seconds": secs})
                return {"stop": "awaiting_signoff", "verdict": "NEEDS_REVIEW", "report": rep, "tracks": tracks,
                        "trace": trace, "reason": ans.get("reason")}
            elif isinstance(ans.get("choice"), int) and 0 <= ans["choice"] < len(cands):
                choice, conf, reason = ans["choice"], ans.get("confidence"), ans.get("reason", "")
        a = cands[choice]
        if a["type"] == "set_identity":
            if ask is None or (conf or 0) < 0.8:
                return {"stop": "awaiting_signoff", "verdict": "NEEDS_REVIEW", "report": rep, "tracks": tracks,
                        "trace": trace, "reason": f"identity of ID {a['id']} uncertain"}
            new_over = dict(overrides); new_over[(a["item"], a["id"])] = a["sku"]
            new_tracks = tracks
        else:
            new_over, new_tracks = overrides, apply(tracks, a)
        new_rep = audit(new_tracks, order, cat_skus, frames_dir, refs, new_over, params)
        before = {(v["check"], v["item"]) for v in rep["violations"] if v["check"] != "C1_order"}
        after = {(v["check"], v["item"]) for v in new_rep["violations"] if v["check"] != "C1_order"}
        n_b = sum(v["check"] != "C1_order" for v in rep["violations"])
        n_a = sum(v["check"] != "C1_order" for v in new_rep["violations"])
        accepted = n_a < n_b and not (after - before)
        trace.append({"step": step, "candidates": [_describe(c, i) for i, c in enumerate(cands)], "choice": choice,
                      "action": _describe(a, choice), "confidence": conf, "reason": reason, "seconds": secs,
                      "accepted": accepted, "integrity_violations": [n_b, n_a]})
        if not accepted:
            return {"stop": "awaiting_signoff", "verdict": "NEEDS_REVIEW", "report": rep, "tracks": tracks,
                    "trace": trace, "reason": "verifier rejected the proposed repair"}
        tracks, overrides, rep = new_tracks, new_over, new_rep
    if rep["verdict"] in ("PASS", "FAIL"):
        return {"stop": "answered", "verdict": rep["verdict"], "report": rep, "tracks": tracks, "trace": trace}
    return {"stop": "paused", "verdict": "NEEDS_REVIEW", "report": rep, "tracks": tracks, "trace": trace}
