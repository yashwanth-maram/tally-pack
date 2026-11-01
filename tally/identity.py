"""Tally Pack - SKU identity: which product is each tracked item?

SAM 3.1 tracks product *categories* reliably ("packet", "can", "bottle") but treats brand
phrases ("noodle packet" vs "chips bag") as the same concept. Identity is therefore resolved
per track by appearance: the track's crops are compared with the reference cut-out of every
SKU in its category (HSV colour histograms on a 2x2 spatial grid). If the best match does not
beat the runner-up by a clear margin, the item is marked AMBIGUOUS and goes to the investigator
instead of being guessed - the verifier never clears a box on a low-confidence identity.
"""
import os

import cv2
import numpy as np

BINS = (18, 4, 4)


def _hist(bgr, mask=None):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    feats = []
    for gy in range(2):
        for gx in range(2):
            ys, xs = slice(gy * h // 2, (gy + 1) * h // 2), slice(gx * w // 2, (gx + 1) * w // 2)
            m = None if mask is None else mask[ys, xs]
            hist = cv2.calcHist([hsv[ys, xs]], [0, 1, 2], m, list(BINS), [0, 180, 0, 256, 0, 256]).flatten()
            feats.append(hist / (hist.sum() + 1e-6))
    return np.concatenate(feats) / 4.0


def reference_descriptors(spr_dir, skus):
    """Descriptor per SKU from its RGBA cut-out (alpha used as the mask)."""
    refs = {}
    for s in skus:
        p = os.path.join(spr_dir, f"{s}.png")
        if os.path.exists(p):
            spr = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            refs[s] = _hist(spr[:, :, :3], (spr[:, :, 3] > 128).astype(np.uint8))
    return refs


def _similarity(a, b):
    return float(np.sum(np.sqrt(a * b)))       # Bhattacharyya coefficient, 1 = identical


def _crop(frame, box, inner=0.7):
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    w2, h2 = w * inner / 2, h * inner / 2
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = int(max(0, cx - w2)), int(max(0, cy - h2)), int(min(W, cx + w2)), int(min(H, cy + h2))
    return frame[y0:y1, x0:x1] if x1 - x0 > 4 and y1 - y0 > 4 else None


def track_descriptor(frames_dir, rec, oid, samples=8):
    """Average descriptor over frames where the item is stationary (start and end of its life)."""
    fs = sorted(f for f, dets in rec.items() if any(d["id"] == oid for d in dets))
    if not fs:
        return None
    k = max(1, len(fs) // 6)
    pick = fs[:k] + fs[-k:]
    pick = pick[:: max(1, len(pick) // samples)][:samples]
    descs = []
    for f in pick:
        det = next(d for d in rec[f] if d["id"] == oid)
        img = cv2.imread(os.path.join(frames_dir, f"{f:05d}.jpg"))
        c = _crop(img, det["box"]) if img is not None else None
        if c is not None:
            descs.append(_hist(c))
    return np.mean(descs, axis=0) if descs else None


def assign(frames_dir, tracks, category_skus, refs, margin=0.04):
    """Map every (category prompt, track id) to a SKU, or to 'AMBIGUOUS:<a>|<b>'.

    category_skus: {category prompt: [skus in that category]}
    Returns {prompt: {id: {"sku", "score", "runner_up", "margin"}}}.
    """
    out = {}
    for prompt, skus in category_skus.items():
        rec = tracks["prompts"].get(prompt, {})
        ids = sorted({d["id"] for dets in rec.values() for d in dets})
        out[prompt] = {}
        for oid in ids:
            desc = track_descriptor(frames_dir, rec, oid)
            if desc is None:
                continue
            scored = sorted(((_similarity(desc, refs[s]), s) for s in skus if s in refs), reverse=True)
            if not scored:
                continue
            best, sku = scored[0]
            second = scored[1][0] if len(scored) > 1 else 0.0
            name = sku if (best - second) >= margin or len(scored) == 1 else f"AMBIGUOUS:{sku}|{scored[1][1]}"
            out[prompt][oid] = {"sku": name, "score": round(best, 3), "runner_up": round(second, 3),
                                "margin": round(best - second, 3)}
    return out


def sku_counts(ledger, assignment):
    """Items that ended inside the container, per SKU (ambiguous ones counted separately)."""
    counts, ambiguous = {}, []
    for prompt, items in ledger["items"].items():
        for oid, it in items.items():
            if it["final_zone"] != "container":
                continue
            a = assignment.get(prompt, {}).get(oid)
            if a is None:
                continue
            if a["sku"].startswith("AMBIGUOUS"):
                ambiguous.append({"prompt": prompt, "id": oid, **a})
            else:
                counts[a["sku"]] = counts.get(a["sku"], 0) + 1
    return counts, ambiguous


def check_order(counts, ambiguous, order):
    """C1 at SKU level. Ambiguous identities can never produce a PASS."""
    violations = []
    for s in sorted(set(order) | set(counts)):
        got, want = counts.get(s, 0), order.get(s, 0)
        if got != want and not ambiguous:
            violations.append({"check": "C1_order", "item": s, "frames": [], "ids": [],
                               "detail": f"{'missing' if got < want else 'extra'}: {got} {s} in box, order says {want}"})
    for a in ambiguous:
        violations.append({"check": "C5_identity_ambiguous", "item": a["prompt"], "frames": [], "ids": [a["id"]],
                           "detail": f"ID {a['id']} could be {a['sku'][10:]} (margin {a['margin']})"})
    return violations
