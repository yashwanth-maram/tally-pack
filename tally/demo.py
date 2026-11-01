"""Tally Pack - demo renderer: annotated packing video + a JSON "story" per case, for the UI and the video.

The cached tracks store boxes only, so SAM 3.1 is re-run once per showcase clip to recover the masks
(deterministic, so detections match the cached tracks; matching is done by box IoU, which also survives
injected ID switches). The Glimmer investigation is replayed from the saved evaluation trace, so building
the demo needs no model calls beyond SAM.
"""
import gzip
import json
import os
import pickle
import subprocess
import textwrap

import cv2
import numpy as np

from . import agent as A
from . import ledger as L

SKU_BGR = {"maggi": (0, 215, 255), "parle_g": (0, 120, 255), "lays": (200, 80, 220),
           "coke_can": (40, 40, 230), "dettol": (80, 190, 60)}
VERDICT_BGR = {"PASS": (60, 170, 30), "FAIL": (40, 40, 210), "NEEDS_REVIEW": (0, 150, 230)}
PANEL_W = 620
FONT = cv2.FONT_HERSHEY_SIMPLEX


# ------------------------------------------------------------------ SAM 3.1 masks
def sam_masks(frames_dir, prompts, scale=0.5, cache=None):
    """{prompt: {frame: [(box_xywh_px, mask_uint8_at_scale)]}}; cached as gzip pickle if `cache` given."""
    if cache and os.path.exists(cache):
        return pickle.load(gzip.open(cache))
    from . import perception as P
    pred = P.get_predictor()
    W, H = P._frame_size(frames_dir)
    sw, sh = int(W * scale), int(H * scale)
    out = {}
    for text in prompts:
        sid = pred.handle_request({"type": "start_session", "resource_path": frames_dir})["session_id"]
        pred.handle_request({"type": "add_prompt", "session_id": sid, "frame_index": 0, "text": text})
        rec = {}
        for o in pred.handle_stream_request({"type": "propagate_in_video", "session_id": sid}):
            dets = []
            for m in o["outputs"]["out_binary_masks"]:
                m = np.asarray(m).squeeze().astype(np.uint8)
                if m.ndim != 2 or not m.any():
                    continue
                m = cv2.resize(m, (sw, sh), interpolation=cv2.INTER_NEAREST)
                ys, xs = np.nonzero(m)
                if len(xs):
                    box = [xs.min() / scale, ys.min() / scale, (np.ptp(xs) + 1) / scale, (np.ptp(ys) + 1) / scale]
                    dets.append((box, m))
            rec[int(o["frame_index"])] = dets
        pred.handle_request({"type": "close_session", "session_id": sid})
        out[text] = rec
    if cache:
        pickle.dump(out, gzip.open(cache, "wb"))
    return out


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    iw, ih = max(0, min(ax2, bx2) - max(ax1, bx1)), max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    return inter / max(1e-6, a[2] * a[3] + b[2] * b[3] - inter)


def _match_mask(masks, prompt, f, box):
    best, bi = None, 0.5
    for mb, m in (masks or {}).get(prompt, {}).get(f, []):
        i = _iou(mb, box)
        if i > bi:
            best, bi = m, i
    return best


# ------------------------------------------------------------------ replay a saved investigation
def replay(trace):
    """An `ask` function that returns the saved Glimmer decisions in order (no model needed)."""
    it = iter(trace or [])

    def ask(system, prompt, images=()):
        t = next(it, None)
        if t is None:
            return None, "", 0.0
        return {"choice": t.get("choice"), "confidence": t.get("confidence"), "reason": t.get("reason")}, "", t.get("seconds", 0.0)
    return ask


def zone_series(led, max_gap):
    """(prompt, id) -> per-frame debounced zone (None when the item is not alive)."""
    n, out = led["n_frames"], {}
    for prompt, items in led["items"].items():
        for oid, it in items.items():
            z, state = [None] * n, None
            for f in range(it["first"], n):
                if f in it["timeline"]:
                    state = it["timeline"][f]
                elif f - it["last"] > max_gap and it["final_zone"] != "container":
                    break
                z[f] = state
            out[(prompt, oid)] = z
    return out


# ------------------------------------------------------------------ drawing
def _text(img, s, xy, scale=0.5, color=(235, 235, 235), thick=1):
    cv2.putText(img, s, xy, FONT, scale, color, thick, cv2.LINE_AA)


def _act(t):
    a = t.get("action") or ""
    return a.split(": ", 1)[1] if ": " in a[:4] else a


def _verdict_lines(story):
    v = story["final"]
    if v == "PASS":
        head = "PASS - box matches the order"
    elif v == "FAIL":
        head = "FAIL - box does not match the order"
    else:
        head = "NEEDS REVIEW - sent to a human"
    lines = []
    if story["first"] == "NEEDS_REVIEW":
        lines.append("Verifier: evidence inconsistent ->")
        for v0 in story["first_violations"][:2]:
            lines += textwrap.wrap(f"  {v0}", 46)
        for t in story["trace"]:
            if t.get("choice") == "escalate":
                w = textwrap.wrap(f"Glimmer: escalate - {t.get('reason') or ''}", 46)
                lines += w[:5] if len(w) <= 5 else w[:4] + [w[4][:42] + " ..."]
            else:
                lines += textwrap.wrap(f"Glimmer: {_act(t)} (conf {t.get('confidence')})", 46)[:2]
                n_b, n_a = (t.get("integrity_violations") or [0, 0])
                lines.append(f"Verifier re-check: {n_b} -> {n_a} issues, "
                             f"{'ACCEPTED' if t.get('accepted') else 'REJECTED'}")
    for d in story["final_details"][:3]:
        lines += textwrap.wrap(d, 46)
    return head, lines


def render(frames_dir, tracks, rep, story, masks, out_mp4, fps=10, hold_s=6.0, S=1.5):
    """Video (upscaled by S for readable text) + side panel: live order counts, ledger, then the verdict."""
    led, n = rep["ledger"], tracks["n_frames"]
    W, H = tracks["size"]
    SW, SH, PW = int(W * S), int(H * S), PANEL_W
    zones = zone_series(led, L.Params().max_gap)
    asg = rep["assignment"]
    cid = led["container_id"]
    events = rep["events"]
    order = story["order"]
    skus = sorted(set(order) | set(story["final_counts"]))
    tmp = out_mp4 + ".raw.mp4"
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (SW + PW, SH))
    head, vlines = _verdict_lines(story)
    for f in list(range(n)) + [n - 1] * int(hold_s * fps):
        hold = f == n - 1
        img = cv2.imread(f"{frames_dir}/{f:05d}.jpg")
        over, zone, labels = img.copy(), None, []
        for d in tracks["prompts"].get(A.CONTAINER, {}).get(f, []):
            if d["id"] == cid:
                zone = [int(v * S) for v in d["box"]]
        for prompt in asg:
            for d in tracks["prompts"].get(prompt, {}).get(f, []):
                sku = asg[prompt].get(d["id"], {}).get("sku", "?")
                col = SKU_BGR.get(sku, (160, 160, 160))
                m = _match_mask(masks, prompt, f, d["box"])
                x, y, w, h = [int(v) for v in d["box"]]
                if m is not None:
                    mm = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
                    over[mm] = col
                    cnts, _ = cv2.findContours(mm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(img, cnts, -1, col, 2)
                    ys, xs = np.nonzero(mm)
                    x, y = int(xs.min()), int(ys.min())
                else:
                    cv2.rectangle(over, (x, y), (x + w, y + h), col, -1)
                    cv2.rectangle(img, (x, y), (x + w, y + h), col, 2)
                name = "??" if sku.startswith("AMBIG") else sku
                labels.append((f"{name} #{d['id']}", (int(x * S), max(22, int(y * S) - 8)), col))
        img = cv2.resize(cv2.addWeighted(over, 0.42, img, 0.58, 0), (SW, SH), interpolation=cv2.INTER_LINEAR)
        if zone:
            x, y, w, h = zone
            x0, y0, x1, y1 = max(3, x), max(3, y), min(SW - 4, x + w), min(SH - 4, y + h)
            cv2.rectangle(img, (x0, y0), (x1, y1), (255, 230, 120), 2)
            _text(img, "PACKING ZONE", (x0 + 10, y0 + 30), 0.75, (0, 0, 0), 5)
            _text(img, "PACKING ZONE", (x0 + 10, y0 + 30), 0.75, (255, 230, 120), 2)
        for s_, xy, col in labels:
            _text(img, s_, xy, 0.75, (0, 0, 0), 5)
            _text(img, s_, xy, 0.75, col, 2)

        # ---- side panel
        p = np.full((SH, PW, 3), 26, np.uint8)
        _text(p, "TALLY PACK", (22, 48), 1.25, (255, 255, 255), 3)
        _text(p, f"t = {f / fps:4.1f}s    SAM 3.1 + ledger", (22, 82), 0.65, (170, 170, 170), 1)
        _text(p, "ORDER", (22, 130), 0.75, (200, 200, 200), 2)
        _text(p, "in box / ordered", (PW - 250, 130), 0.65, (170, 170, 170), 1)
        live = {}
        for (prompt, oid), z in zones.items():
            if z[f] == "container":
                sk = asg.get(prompt, {}).get(oid, {}).get("sku", "?")
                live[sk] = live.get(sk, 0) + 1
        y = 170
        for sk in skus:
            got, want = (story["final_counts"].get(sk, 0) if hold else live.get(sk, 0)), order.get(sk, 0)
            col = (90, 215, 90) if got == want else ((235, 235, 235) if not hold else (80, 80, 240))
            cv2.circle(p, (32, y - 9), 10, SKU_BGR.get(sk, (160, 160, 160)), -1)
            _text(p, sk, (54, y), 0.85, col, 2)
            _text(p, f"{got} / {want}", (PW - 150, y), 0.85, col, 2)
            y += 40
        y += 14
        if not hold:
            _text(p, "LEDGER", (22, y), 0.75, (200, 200, 200), 2)
            y += 36
            for e in [e for e in events if e["f"] <= f][-8:]:
                sk = asg.get(e["item"], {}).get(e["id"], {}).get("sku", e["item"])
                _text(p, f"{e['t']:4.1f}s  {sk} #{e['id']}  {e['event']}", (22, y), 0.62, (230, 230, 230), 1)
                y += 30
        else:
            col = VERDICT_BGR[story["final"]]
            cv2.rectangle(p, (14, y - 10), (PW - 14, y + 50), col, -1)
            _text(p, head, (26, y + 30), 0.72, (255, 255, 255), 2)
            y += 86
            for ln in vlines[: max(0, (SH - y - 10) // 28)]:
                _text(p, ln, (22, y), 0.6, (235, 235, 235), 1)
                y += 28
            cv2.rectangle(img, (0, 0), (SW - 1, SH - 1), col, 12)
        vw.write(np.hstack([img, p]))
    vw.release()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264", "-preset", "veryfast",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_mp4], check=True)
    os.remove(tmp)
    return out_mp4


# ------------------------------------------------------------------ one showcase case
def build_case(gt, kind, tracks, frames_dir, refs, cat_all, row, out_dir, masks=None, inject=None):
    """Re-audit (no model calls), replay the saved investigation, render the video, write the story JSON."""
    tr = tracks
    if kind == "injected":
        tr = inject(tracks)
    rep = A.audit(tr, gt["order"], cat_all, frames_dir, refs)
    if rep["verdict"] == "NEEDS_REVIEW":
        r = A.investigate(tr, gt["order"], cat_all, frames_dir, refs, ask=replay(row.get("trace")))
        final_rep, final, stop, trace = r["report"], r["verdict"], r["stop"], r["trace"]
    else:
        final_rep, final, stop, trace = rep, rep["verdict"], "answered", []
    key = f"{gt['name']}__{kind}"
    story = {"key": key, "name": gt["name"], "kind": kind, "scenario": gt["scenario"], "order": gt["order"],
             "expected": gt["expected"], "first": rep["verdict"], "final": final, "stop": stop,
             "first_violations": [v["detail"] for v in rep["violations"] if v["check"] != "C1_order"],
             "final_details": [v["detail"] for v in final_rep["violations"]],
             "final_counts": final_rep["counts"], "events": final_rep["events"],
             "trace": [{k: t.get(k) for k in ("step", "choice", "action", "confidence", "reason", "seconds",
                                              "accepted", "integrity_violations", "candidates")} for t in trace],
             "glimmer_seconds": round(sum(t.get("seconds") or 0 for t in trace), 1)}
    os.makedirs(out_dir, exist_ok=True)
    story["video"] = render(frames_dir, tr, rep, story, masks, f"{out_dir}/{key}.mp4", fps=tracks["fps"])
    json.dump(story, open(f"{out_dir}/{key}.json", "w"), indent=1, default=str)
    return story
