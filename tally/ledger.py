"""Tally Pack - ledger + verifier (zero model calls).

Turns SAM 3.1 tracks into a per-item ledger of zone states and events, then checks
deterministic invariants:

  C1 order       items that ended in the container == what the order asked for
  C2 conservation  number of items alive (seen recently, or sealed in the container)
                 stays equal to the starting count -> catches lost / duplicated IDs
  C3 continuity  no item jumps across the frame between consecutive sightings
  C4 identity    an ID that appears mid-clip right where another ID just vanished
                 is a suspected ID switch (the case the investigator repairs)

Every violation carries a frame window so the investigator (Muse Wonder / Glimmer)
only looks at the seconds that matter.
"""
from dataclasses import dataclass, field, asdict


@dataclass
class Params:
    debounce: int = 3          # frames a new zone must persist before a state change counts
    max_gap: int = 15          # frames an item may be unseen (e.g. under a hand) and still count as alive
    warmup: int = 5            # frames used to establish the starting inventory
    edge: float = 0.04         # fraction of frame treated as the border (items may legitimately enter/leave there)
    jump: float = 0.25         # max centroid jump per frame, as a fraction of frame width
    switch_radius: float = 0.08  # vanish->appear within this distance (fraction of width) = suspected ID switch
    pad: float = 0.04          # padding added around the container box (fraction of frame width)


@dataclass
class Violation:
    check: str
    item: str
    frames: list
    detail: str
    ids: list = field(default_factory=list)


def main_container(rec):
    """The container is the tracked object with the largest total area over the clip."""
    score = {}
    for dets in rec.values():
        for d in dets:
            score[d["id"]] = score.get(d["id"], 0) + d["area"] + 1
    return max(score, key=score.get) if score else None


def container_boxes(rec, cid, n, W, pad):
    """Per-frame container box (padded). If the container is briefly unseen, keep its last box."""
    out, last = {}, None
    for f in range(n):
        for d in rec.get(f, []):
            if d["id"] == cid:
                x, y, w, h = d["box"]
                p = pad * W
                last = (x - p, y - p, x + w + p, y + h + p)
        out[f] = last
    return out


def _inside(c, box):
    return box is not None and box[0] <= c[0] <= box[2] and box[1] <= c[1] <= box[3]


def _near_edge(c, W, H, e):
    return c[0] < e * W or c[0] > (1 - e) * W or c[1] < e * H or c[1] > (1 - e) * H


def build_ledger(tracks, container_prompt, item_prompts, params=None):
    """Return a ledger dict: per item prompt, per ID, the debounced zone timeline and events."""
    P = params or Params()
    n, (W, H), fps = tracks["n_frames"], tracks["size"], tracks["fps"]
    crec = tracks["prompts"][container_prompt]
    cid = main_container(crec)
    cbox = container_boxes(crec, cid, n, W, P.pad)

    ledger = {"fps": fps, "n_frames": n, "size": [W, H], "container_id": cid, "items": {}}
    for prompt in item_prompts:
        rec = tracks["prompts"][prompt]
        obs = {}  # id -> list of (frame, centroid, raw_zone)
        for f in range(n):
            for d in rec.get(f, []):
                z = "container" if _inside(d["c"], cbox[f]) else "outside"
                obs.setdefault(d["id"], []).append((f, d["c"], z))

        items = {}
        for oid, seq in obs.items():
            state, cand, run, events, timeline = None, None, 0, [], {}
            for f, c, z in seq:
                if state is None:
                    state = z
                    events.append({"f": f, "type": "appeared", "zone": z, "c": c})
                elif z != state:
                    run = run + 1 if z == cand else 1
                    cand = z
                    if run >= P.debounce:
                        events.append({"f": f, "type": "placed" if z == "container" else "removed", "zone": z, "c": c})
                        state, cand, run = z, None, 0
                else:
                    cand, run = None, 0
                timeline[f] = state
            last_f, last_c = seq[-1][0], seq[-1][1]
            if last_f < n - 1 - P.max_gap:
                events.append({"f": last_f, "type": "sealed" if state == "container" else "vanished",
                               "zone": state, "c": last_c})
            items[oid] = {"first": seq[0][0], "last": last_f, "first_c": seq[0][1], "last_c": last_c,
                          "final_zone": state, "events": events, "timeline": timeline,
                          "path": [(f, c) for f, c, _ in seq]}
        ledger["items"][prompt] = items
    return ledger


def verify(ledger, order, params=None):
    """Check invariants. `order` maps item prompt -> expected count in the container at the end.

    Returns {"verdict", "violations", "counts", "model_calls": 0}.
    """
    P = params or Params()
    n, (W, H) = ledger["n_frames"], ledger["size"]
    violations, counts = [], {}

    for prompt, items in ledger["items"].items():
        # starting inventory = IDs first seen during warm-up
        start_ids = {i for i, it in items.items() if it["first"] < P.warmup}
        n0 = len(start_ids)

        # C1 order check
        in_box = sorted(i for i, it in items.items() if it["final_zone"] == "container")
        expected = order.get(prompt)
        counts[prompt] = {"start": n0, "in_container": len(in_box), "expected": expected, "ids_in_container": in_box}
        if expected is not None and len(in_box) != expected:
            kind = "missing" if len(in_box) < expected else "extra"
            violations.append(Violation("C1_order", prompt, [n - 1, n - 1],
                                        f"{kind}: {len(in_box)} in container, order says {expected}", in_box))

        # C2 conservation: alive = seen within max_gap, or sealed inside the container
        bad_run, bad_start, worst = 0, None, None
        for f in range(P.warmup, n):
            alive = 0
            for it in items.values():
                if it["first"] > f:
                    continue
                if f - it["last"] <= P.max_gap or it["final_zone"] == "container":
                    alive += 1
            if alive != n0:
                if bad_run == 0:
                    bad_start = f
                bad_run += 1
                worst = alive
            else:
                if bad_run > P.debounce:
                    violations.append(Violation("C2_conservation", prompt, [bad_start, f - 1],
                                                f"{worst} items alive, started with {n0}"))
                bad_run = 0
        if bad_run > P.debounce:
            violations.append(Violation("C2_conservation", prompt, [bad_start, n - 1],
                                        f"{worst} items alive, started with {n0}"))

        # C3 continuity: no teleporting between consecutive sightings
        for oid, it in items.items():
            pts = it["path"]
            for (f1, c1), (f2, c2) in zip(pts, pts[1:]):
                gap = max(1, f2 - f1)
                dist = ((c2[0] - c1[0]) ** 2 + (c2[1] - c1[1]) ** 2) ** 0.5
                if gap <= 2 and dist > P.jump * W:
                    violations.append(Violation("C3_continuity", prompt, [f1, f2],
                                                f"ID {oid} jumped {dist:.0f}px in {gap} frame(s)", [oid]))

        # C4 identity: a mid-clip appearance next to a recent vanish = suspected ID switch
        vanished = [(i, it) for i, it in items.items() if any(e["type"] == "vanished" for e in it["events"])]
        for oid, it in items.items():
            if it["first"] < P.warmup or _near_edge(it["first_c"], W, H, P.edge):
                continue
            match = None
            for vid, vt in vanished + [(i, t) for i, t in items.items() if i != oid]:
                if vid == oid or not (0 <= it["first"] - vt["last"] <= P.max_gap):
                    continue
                d = ((it["first_c"][0] - vt["last_c"][0]) ** 2 + (it["first_c"][1] - vt["last_c"][1]) ** 2) ** 0.5
                if d <= P.switch_radius * W:
                    match = vid
                    break
            detail = (f"ID {oid} appeared at f{it['first']} where ID {match} disappeared -> suspected ID switch"
                      if match is not None else f"ID {oid} appeared mid-clip away from the frame edge")
            violations.append(Violation("C4_identity", prompt, [max(0, it["first"] - P.max_gap), it["first"] + P.debounce],
                                        detail, [oid] + ([match] if match is not None else [])))

    return {"verdict": "PASS" if not violations else "FAIL",
            "violations": [asdict(v) for v in violations], "counts": counts, "model_calls": 0}


def events_table(ledger):
    """Flat, time-ordered list of events for the UI and the investigator prompt."""
    rows = []
    fps = ledger["fps"]
    for prompt, items in ledger["items"].items():
        for oid, it in items.items():
            for e in it["events"]:
                rows.append({"t": round(e["f"] / fps, 2), "f": e["f"], "item": prompt, "id": oid,
                             "event": e["type"], "zone": e["zone"]})
    return sorted(rows, key=lambda r: r["f"])


def merge_ids(tracks, prompt, keep, drop):
    """Repair action: relabel every detection of ID `drop` as ID `keep` (an ID switch fix).
    Returns a new tracks dict; the original is untouched so the repair can be rejected."""
    import copy
    t = copy.deepcopy(tracks)
    for dets in t["prompts"][prompt].values():
        ids = {d["id"] for d in dets}
        for d in dets:
            if d["id"] == drop and keep not in ids:
                d["id"] = keep
    return t


def try_repair(tracks, container_prompt, item_prompts, order, action, params=None):
    """Apply a proposed repair and accept it only if the verifier passes the checks it touched
    without introducing new violations. This is the gate the investigator cannot skip."""
    before = verify(build_ledger(tracks, container_prompt, item_prompts, params), order, params)
    if action["type"] == "merge":
        cand = merge_ids(tracks, action["item"], action["keep"], action["drop"])
    else:
        return {"accepted": False, "reason": f"unknown action {action['type']}", "before": before, "after": before}
    after = verify(build_ledger(cand, container_prompt, item_prompts, params), order, params)
    accepted = len(after["violations"]) < len(before["violations"]) and not (
        {(v["check"], v["item"]) for v in after["violations"]} - {(v["check"], v["item"]) for v in before["violations"]})
    return {"accepted": accepted, "tracks": cand if accepted else tracks, "before": before, "after": after}
