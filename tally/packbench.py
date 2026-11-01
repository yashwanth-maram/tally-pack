"""Tally Pack - PackBench: fixed-camera packing clips with ground truth by construction.

Real assets only: a real table + open box (Pexels 7855142), a real hand cut-out (same clip),
real product photos (Open Food Facts) cut out with SAM 3.1. The generator animates a hand
carrying products from the table into the box and injects controlled packing errors.

Scenarios
  clean      pack exactly the order                         -> PASS
  missing    one ordered item is left on the table          -> FAIL (missing)
  extra      one item that was not ordered is also packed   -> FAIL (extra)
  wrong      a look-alike substitute is packed instead      -> FAIL (missing + extra)
  occlusion  clean, but the hand fully covers an item       -> PASS (robustness)
  twins      clean, two identical items cross under a hand  -> PASS (ID-switch stress)
"""
import json
import os
import random

import cv2
import numpy as np

# SKU -> SAM text prompt used to track it (the contract compiler will produce these later)
SKUS = {
    "maggi": "noodle packet",
    "parle_g": "biscuit packet",
    "lays": "chips bag",
    "coke_can": "soda can",
    "dettol": "plastic bottle",
}
# SKU -> product category: SAM 3.1 tracks categories reliably; brand identity is resolved by appearance
CATEGORY = {"maggi": "packet", "parle_g": "packet", "lays": "packet", "coke_can": "can", "dettol": "bottle"}
SCENARIOS = ["clean", "missing", "extra", "wrong", "occlusion", "twins"]


def load_assets(spr_dir, item_px=95, hand_px=300):
    def rd(name, px):
        s = cv2.imread(os.path.join(spr_dir, name), cv2.IMREAD_UNCHANGED)
        k = px / max(s.shape[:2])
        return cv2.resize(s, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    items = {k: rd(f"{k}.png", item_px) for k in SKUS if os.path.exists(os.path.join(spr_dir, f"{k}.png"))}
    return {"bg": cv2.imread(os.path.join(spr_dir, "background.jpg")), "hand": rd("hand.png", hand_px), "items": items}


def paste(frame, spr, cx, cy, shadow=True):
    """Alpha-blend a BGRA sprite centred at (cx, cy), with a soft drop shadow."""
    h, w = spr.shape[:2]
    x0, y0 = int(cx - w / 2), int(cy - h / 2)
    H, W = frame.shape[:2]
    for dx, dy, layer in (((8, 10, "shadow"),) if shadow else ()) + ((0, 0, "sprite"),):
        xa, ya = max(0, x0 + dx), max(0, y0 + dy)
        xb, yb = min(W, x0 + dx + w), min(H, y0 + dy + h)
        if xa >= xb or ya >= yb:
            continue
        sx, sy = xa - (x0 + dx), ya - (y0 + dy)
        a = spr[sy:sy + yb - ya, sx:sx + xb - xa, 3:4].astype(np.float32) / 255.0
        roi = frame[ya:yb, xa:xb].astype(np.float32)
        if layer == "shadow":
            a = cv2.GaussianBlur(a, (21, 21), 0)[..., None] * 0.45
            frame[ya:yb, xa:xb] = (roi * (1 - a)).astype(np.uint8)
        else:
            frame[ya:yb, xa:xb] = (a * spr[sy:sy + yb - ya, sx:sx + xb - xa, :3] + (1 - a) * roi).astype(np.uint8)
    return frame


def _ease(t):
    return t * t * (3 - 2 * t)


def plan(scenario, rng, skus):
    """Choose the order, what is laid out on the table, and what actually gets packed."""
    pool = list(skus)
    rng.shuffle(pool)
    k = rng.randint(3, 4)
    order_list = pool[:k]
    if scenario == "twins":
        order_list = [order_list[0]] + order_list[:k - 1]          # one SKU ordered twice
    order = {s: order_list.count(s) for s in order_list}
    table = list(order_list)
    packed = list(order_list)
    note = ""
    spare = [s for s in pool if s not in order]
    if scenario == "missing":
        drop = rng.choice(order_list)
        packed.remove(drop)
        note = f"left {drop} on the table"
    elif scenario == "extra":
        add = spare[0] if spare else rng.choice(order_list)
        table.append(add)
        packed.append(add)
        note = f"also packed {add}"
    elif scenario == "wrong":
        sub = spare[0] if spare else rng.choice(order_list)
        victim = rng.choice(order_list)
        table.append(sub)
        packed.remove(victim)
        packed.append(sub)
        note = f"packed {sub} instead of {victim}"
    return order, table, packed, note


def make_clip(out_dir, scenario, seed, assets, interior, table_zone, fps=10, pick_s=2.6):
    """Render one PackBench clip as numbered JPEGs (+ preview mp4) and return its ground truth."""
    rng = random.Random(seed)
    bg, hand, sprites = assets["bg"], assets["hand"], assets["items"]
    H, W = bg.shape[:2]
    order, table, packed, note = plan(scenario, rng, list(sprites))

    # lay items out on the table zone (left strip), spaced vertically
    tx0, ty0, tx1, ty1 = table_zone
    n_t = len(table)
    ys = np.linspace(ty0 + 55, ty1 - 55, n_t)
    if scenario == "twins":   # put the twin pair close together so their paths/cut-outs interact
        twin = [i for i, s in enumerate(table) if table.count(s) == 2]
        order_idx = twin + [i for i in range(n_t) if i not in twin]
        table = [table[i] for i in order_idx]
    items = [{"sku": s, "pos": [rng.uniform(tx0 + 55, tx1 - 55), float(y)], "packed": False} for s, y in zip(table, ys)]
    if scenario == "twins":
        items[1]["pos"] = [items[0]["pos"][0] + 40, items[0]["pos"][1] + 70]

    # box drop targets: a grid inside the interior
    ix0, iy0, ix1, iy1 = interior
    grid = [(ix0 + (ix1 - ix0) * (0.18 + 0.32 * (j % 3)), iy0 + (iy1 - iy0) * (0.27 + 0.48 * (j // 3))) for j in range(6)]
    rng.shuffle(grid)

    # which table items get carried, in what order
    to_pack, remaining = [], list(packed)
    for i, it in enumerate(items):
        if it["sku"] in remaining:
            remaining.remove(it["sku"])
            to_pack.append(i)
    rng.shuffle(to_pack)
    if scenario == "twins":   # carry the far twin first so it passes over its sibling
        tw = [i for i in to_pack if items[i]["sku"] == items[0]["sku"]]
        to_pack = sorted(tw, reverse=True) + [i for i in to_pack if i not in tw]

    occl_pick = rng.randrange(len(to_pack)) if scenario == "occlusion" else -1
    hw, hh = hand.shape[1], hand.shape[0]
    home = (W + hw * 0.3, H + hh * 0.2)       # hand rests off-frame, bottom-right
    frames, events, truth = [], [], []

    def render(carried=None, hand_at=None, cover=False):
        fr = bg.copy()
        boxes = []
        for j, it in enumerate(items):
            sh, sw = sprites[it["sku"]].shape[:2]
            boxes.append({"idx": j, "sku": it["sku"], "c": [round(it["pos"][0], 1), round(it["pos"][1], 1)],
                          "box": [round(it["pos"][0] - sw / 2, 1), round(it["pos"][1] - sh / 2, 1), sw, sh],
                          "covered": bool(cover and j == carried)})
        truth.append(boxes)
        for j, it in enumerate(items):
            if j != carried:
                paste(fr, sprites[it["sku"]], *it["pos"])
        if carried is not None:
            paste(fr, sprites[items[carried]["sku"]], *items[carried]["pos"])
        if hand_at is not None:
            off = (0.0, 0.0) if cover else (hw * 0.32, -hh * 0.05)
            paste(fr, hand, hand_at[0] + off[0], hand_at[1] + off[1])
        noise = np.random.default_rng(seed * 1000 + len(frames)).normal(0, 2.0, fr.shape)
        frames.append(np.clip(fr.astype(np.float32) + noise, 0, 255).astype(np.uint8))

    for _ in range(int(fps * 1.0)):
        render()
    n_reach, n_carry, n_rel = int(fps * pick_s * 0.3), int(fps * pick_s * 0.5), int(fps * pick_s * 0.2)
    for p, i in enumerate(to_pack):
        it = items[i]
        start = list(it["pos"])
        target = grid[p % len(grid)]
        cover = p == occl_pick
        for f in range(n_reach):                       # hand reaches for the item
            t = _ease((f + 1) / n_reach)
            render(hand_at=(home[0] + (start[0] - home[0]) * t, home[1] + (start[1] - home[1]) * t))
        for f in range(n_carry):                       # hand carries it into the box
            t = _ease((f + 1) / n_carry)
            it["pos"] = [start[0] + (target[0] - start[0]) * t, start[1] + (target[1] - start[1]) * t]
            render(carried=i, hand_at=it["pos"], cover=cover and 0.2 < t < 0.85)
        it["packed"] = True
        events.append({"f": len(frames) - 1, "sku": it["sku"], "event": "placed"})
        for f in range(n_rel):                         # hand withdraws
            t = _ease((f + 1) / n_rel)
            render(hand_at=(it["pos"][0] + (home[0] - it["pos"][0]) * t, it["pos"][1] + (home[1] - it["pos"][1]) * t))
    for _ in range(int(fps * 1.5)):
        render()

    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(out_dir):
        if f.endswith(".jpg"):
            os.remove(os.path.join(out_dir, f))
    for k, fr in enumerate(frames):
        cv2.imwrite(os.path.join(out_dir, f"{k:05d}.jpg"), fr, [cv2.IMWRITE_JPEG_QUALITY, 92])

    in_box = {}
    for it in items:
        if it["packed"]:
            in_box[it["sku"]] = in_box.get(it["sku"], 0) + 1
    kinds = []
    for s in set(order) | set(in_box):
        if in_box.get(s, 0) < order.get(s, 0):
            kinds.append(f"missing:{s}")
        elif in_box.get(s, 0) > order.get(s, 0):
            kinds.append(f"extra:{s}")
    gt = {"scenario": scenario, "seed": seed, "order": order, "in_box": in_box,
          "expected": "PASS" if not kinds else "FAIL", "errors": sorted(kinds), "note": note,
          "n_frames": len(frames), "fps": fps, "size": [W, H], "events": events, "truth": truth,
          "categories": {s: CATEGORY[s] for s in {it["sku"] for it in items}},
          "prompts": {s: SKUS[s] for s in set(order) | set(in_box) | {it["sku"] for it in items}}}
    json.dump(gt, open(os.path.join(out_dir, "gt.json"), "w"), indent=1)
    return gt


def write_mp4(frames_dir, path, fps=10):
    fs = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))
    first = cv2.imread(os.path.join(frames_dir, fs[0]))
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (first.shape[1], first.shape[0]))
    for f in fs:
        vw.write(cv2.imread(os.path.join(frames_dir, f)))
    vw.release()


def build(root_frames, spr_dir, interior, table_zone, n_per=2, seed0=7, mp4_dir=None):
    """Generate n_per clips for every scenario. Returns list of ground-truth dicts."""
    assets = load_assets(spr_dir)
    gts = []
    for si, sc in enumerate(SCENARIOS):
        for r in range(n_per):
            name = f"pb_{sc}_{r}"
            d = os.path.join(root_frames, name)
            gt = make_clip(d, sc, seed0 + 100 * si + r, assets, interior, table_zone)
            gt["name"] = name
            json.dump(gt, open(os.path.join(d, "gt.json"), "w"), indent=1)
            if mp4_dir:
                os.makedirs(mp4_dir, exist_ok=True)
                write_mp4(d, os.path.join(mp4_dir, f"{name}.mp4"))
            gts.append(gt)
    return gts


def contact_sheet(frames_dir, path, k=6, width=320):
    fs = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))
    idx = np.linspace(0, len(fs) - 1, k).astype(int)
    tiles = []
    for i in idx:
        im = cv2.imread(os.path.join(frames_dir, fs[i]))
        im = cv2.resize(im, (width, int(width * im.shape[0] / im.shape[1])))
        cv2.putText(im, f"f{i}", (6, 22), 0, 0.7, (0, 255, 255), 2)
        tiles.append(im)
    cv2.imwrite(path, np.hstack(tiles))
