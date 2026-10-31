"""Tally Pack - PackBench asset harvesting (public sources only).

Products : Open Food Facts front-of-pack photos (CC BY-SA 3.0, attribution kept in sprites/meta.json)
Hand     : cut from Pexels clip 7855142 (Pexels licence, Kampus Production)
Backdrop : first frame of Pexels clip 7855142 (open cardboard box on a table)

Every cut-out is made with SAM 3.1 itself (text prompt -> mask -> RGBA sprite).
"""
import json
import os
import time

import cv2
import numpy as np
import requests

from . import perception as P

OFF_SEARCH = "https://world.openfoodfacts.org/cgi/search.pl"
UA = {"User-Agent": "TallyPack/0.1 (Meta AI hackathon; github.com/yashwanth-maram/tally-pack)"}

# name -> (Open Food Facts search term, SAM prompt used to cut the product out of its photo)
PRODUCTS = {
    "maggi": ("maggi noodles masala", "noodle packet"),
    "parle_g": ("parle-g biscuits", "biscuit packet"),
    "dairy_milk": ("cadbury dairy milk", "chocolate bar"),
    "coke_can": ("coca-cola can", "can"),
    "lays": ("lays chips", "chips bag"),
    "dettol": ("dettol soap", "soap box"),
}


def _fetch_image(url, min_side):
    try:
        raw = requests.get(url, headers=UA, timeout=60).content
    except requests.RequestException:
        return None
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    return img if img is not None and min(img.shape[:2]) >= min_side else None


def _get_json(url, params, tries=3, wait=4):
    for k in range(tries):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=60)
            if r.status_code == 200:
                return r.json()
        except (requests.RequestException, ValueError):
            pass
        time.sleep(wait * (k + 1))
    return None


def _off_candidates(term):
    """Open Food Facts: new search service first, then the classic search endpoint."""
    js = _get_json("https://search.openfoodfacts.org/search", {"q": term, "page_size": 20})
    for p in (js or {}).get("hits", []):
        url = p.get("image_front_url") or p.get("image_url")
        if url:
            yield url, {"source": "Open Food Facts", "code": p.get("code"), "name": p.get("product_name"),
                        "license": "CC BY-SA 3.0 (Open Food Facts)"}
    js = _get_json(OFF_SEARCH, {"search_terms": term, "search_simple": 1, "action": "process", "json": 1,
                                "page_size": 20, "fields": "code,product_name,brands,image_front_url"})
    for p in (js or {}).get("products", []):
        if p.get("image_front_url"):
            yield p["image_front_url"], {"source": "Open Food Facts", "code": p.get("code"),
                                         "name": p.get("product_name"), "license": "CC BY-SA 3.0 (Open Food Facts)"}


def _commons_candidates(term):
    """Wikimedia Commons file search; licence recorded per image."""
    js = _get_json("https://commons.wikimedia.org/w/api.php", {
        "action": "query", "format": "json", "generator": "search", "gsrnamespace": 6, "gsrsearch": term,
        "gsrlimit": 10, "prop": "imageinfo", "iiprop": "url|extmetadata", "iiurlwidth": 800})
    pages = sorted(((js or {}).get("query") or {}).get("pages", {}).values(), key=lambda p: p.get("index", 0))
    for p in pages:
        ii = (p.get("imageinfo") or [{}])[0]
        url = ii.get("thumburl") or ii.get("url")
        if url and url.lower().split("?")[0].endswith((".jpg", ".jpeg", ".png")):
            lic = ((ii.get("extmetadata") or {}).get("LicenseShortName") or {}).get("value", "see source page")
            yield url, {"source": "Wikimedia Commons", "name": p.get("title"), "page": ii.get("descriptionurl"),
                        "license": lic}


def off_product_image(term, min_side=300):
    """First usable public product photo for `term` (Open Food Facts, then Wikimedia Commons)."""
    for gen in (_off_candidates, _commons_candidates):
        try:
            for url, info in gen(term):
                img = _fetch_image(url, min_side)
                if img is not None:
                    info.update({"term": term, "url": url})
                    return img, info
        except Exception:
            continue
    return None, None


def _masks_from(outputs):
    o = outputs or {}
    probs = o.get("out_probs", [])
    masks = o.get("out_binary_masks", [])
    return [(float(p), np.asarray(m).squeeze().astype(np.uint8)) for p, m in zip(probs, masks)]


def segment_image(img, text, work="/content/_seg", copies=3):
    """Single-image segmentation with the SAM 3.1 video predictor.
    The image is written as a short 3-frame clip (the 1-frame case returns no masks).
    Returns the most confident mask (bool HxW) or None."""
    os.makedirs(work, exist_ok=True)
    for f in os.listdir(work):
        os.remove(os.path.join(work, f))
    for i in range(copies):
        cv2.imwrite(os.path.join(work, f"{i:05d}.jpg"), img)
    pred = P.get_predictor()
    sid = pred.handle_request({"type": "start_session", "resource_path": work})["session_id"]
    resp = pred.handle_request({"type": "add_prompt", "session_id": sid, "frame_index": 0, "text": text})
    cands = _masks_from((resp or {}).get("outputs"))
    for out in pred.handle_stream_request({"type": "propagate_in_video", "session_id": sid}):
        if int(out["frame_index"]) == 0:
            cands += _masks_from(out["outputs"])
    pred.handle_request({"type": "close_session", "session_id": sid})
    best, best_score = None, 0.0
    for p, m in cands:
        if m.ndim != 2 or m.sum() == 0:
            continue
        score = p * (m.sum() ** 0.5)
        if score > best_score:
            best, best_score = m, score
    if best is None:
        return None
    return cv2.resize(best, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)


def cutout(img, mask, feather=2):
    """Crop to the mask and return a BGRA sprite with a softened alpha edge."""
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    alpha = (mask[y0:y1, x0:x1] * 255).astype(np.uint8)
    if feather:
        alpha = cv2.GaussianBlur(alpha, (2 * feather + 1, 2 * feather + 1), 0)
    return np.dstack([img[y0:y1, x0:x1], alpha])


def harvest(out_dir, frames_dir, products=PRODUCTS, hand_frame=30, max_side=400):
    """Build sprites/*.png, background.jpg and meta.json. Returns a one-line summary."""
    os.makedirs(out_dir, exist_ok=True)
    meta = {"products": {}, "hand": None, "background": None}
    ok, failed = [], []
    for name, (term, prompt) in products.items():
        img, info = off_product_image(term)
        if img is None:
            failed.append(name)
            continue
        s = max_side / max(img.shape[:2])
        img = cv2.resize(img, None, fx=s, fy=s) if s < 1 else img
        os.makedirs(os.path.join(out_dir, "raw"), exist_ok=True)
        cv2.imwrite(os.path.join(out_dir, "raw", f"{name}.jpg"), img)
        mask = segment_image(img, prompt) if prompt else None
        if mask is None or mask.mean() < 0.05:
            mask = segment_image(img, "product")
        if mask is None or mask.mean() < 0.05:
            failed.append(f"{name}(src={info['source']},mask={0 if mask is None else round(float(mask.mean()), 3)})")
            continue
        cv2.imwrite(os.path.join(out_dir, f"{name}.png"), cutout(img, mask))
        info["sam_prompt"] = prompt
        meta["products"][name] = info
        ok.append(name)

    frames = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))
    bg = cv2.imread(os.path.join(frames_dir, frames[0]))
    cv2.imwrite(os.path.join(out_dir, "background.jpg"), bg)
    meta["background"] = {"source": "Pexels video 7855142 (Kampus Production)", "frame": frames[0],
                          "license": "Pexels licence"}
    hf = cv2.imread(os.path.join(frames_dir, frames[min(hand_frame, len(frames) - 1)]))
    hmask = segment_image(hf, "hand")
    if hmask is not None and hmask.mean() > 0.005:
        cv2.imwrite(os.path.join(out_dir, "hand.png"), cutout(hf, hmask))
        meta["hand"] = {"source": "Pexels video 7855142 (Kampus Production)", "frame": frames[hand_frame],
                        "license": "Pexels licence"}
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    return f"products ok={ok} failed={failed} | hand={'ok' if meta['hand'] else 'missing'} | background ok"


def contact_sheet(out_dir, path):
    """One image showing every sprite on a grey tile, for a visual check."""
    tiles = []
    for f in sorted(os.listdir(out_dir)):
        if f.endswith(".png"):
            s = cv2.imread(os.path.join(out_dir, f), cv2.IMREAD_UNCHANGED)
            k = 220 / max(s.shape[:2])
            s = cv2.resize(s, None, fx=k, fy=k)
            tile = np.full((240, 240, 3), 128, np.uint8)
            y, x = (240 - s.shape[0]) // 2, (240 - s.shape[1]) // 2
            a = s[:, :, 3:4] / 255.0
            tile[y:y + s.shape[0], x:x + s.shape[1]] = (a * s[:, :, :3] + (1 - a) * tile[y:y + s.shape[0], x:x + s.shape[1]]).astype(np.uint8)
            cv2.putText(tile, f[:-4], (5, 230), 0, 0.6, (255, 255, 255), 2)
            tiles.append(tile)
    if tiles:
        cv2.imwrite(path, np.hstack(tiles))
    return len(tiles)
