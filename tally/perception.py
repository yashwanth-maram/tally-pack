"""Tally Pack - perception layer.

Wraps SAM 3.1 (open weights, facebook/sam3.1) for text-prompted segmentation and
tracking with persistent object IDs. Output is a compact, JSON-serialisable track
record that the ledger and verifier consume (no masks kept by default).
"""
import json
import logging
import os
import sys

import cv2
import numpy as np

SAM3_DIR = os.environ.get("SAM3_DIR", "/content/sam3")
if SAM3_DIR not in sys.path:
    sys.path.insert(0, SAM3_DIR)

_PRED = None


def quiet():
    """Silence SAM 3 INFO logs and progress bars so notebook output stays short."""
    logging.getLogger().setLevel(logging.WARNING)
    for name in list(logging.root.manager.loggerDict):
        logging.getLogger(name).setLevel(logging.WARNING)
    try:
        from functools import partialmethod
        import tqdm
        tqdm.tqdm.__init__ = partialmethod(tqdm.tqdm.__init__, disable=True)
    except Exception:
        pass


def _patch_init_state(model):
    """Work around upstream bug (sam3 commit f1647e8): start_session passes
    offload_state_to_cpu, which the SAM 3.1 multiplex init_state does not accept."""
    while "init_state" in vars(model):
        del model.init_state
    real = type(model).init_state.__get__(model)

    def patched(*args, offload_state_to_cpu=None, **kwargs):
        return real(*args, **kwargs)

    model.init_state = patched


def get_predictor():
    """Build SAM 3.1 once per session. FA3 is off because it needs Hopper GPUs (A100 is Ampere)."""
    global _PRED
    if _PRED is None:
        from sam3.model_builder import build_sam3_predictor
        _PRED = build_sam3_predictor(version="sam3.1", use_fa3=False, compile=False)
        _patch_init_state(_PRED.model)
        quiet()
    return _PRED


def extract_frames(video_path, out_dir, fps_out=10, width=960):
    """Decode a video into numbered JPEGs at ~fps_out, resized to `width` px. Returns (n_frames, fps_out)."""
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(out_dir):
        if f.endswith(".jpg"):
            os.remove(os.path.join(out_dir, f))
    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    step = max(1, round(src_fps / fps_out))
    i = k = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            h = int(frame.shape[0] * width / frame.shape[1])
            cv2.imwrite(os.path.join(out_dir, f"{k:05d}.jpg"), cv2.resize(frame, (width, h)))
            k += 1
        i += 1
    cap.release()
    return k, src_fps / step


def _frame_size(frames_dir):
    first = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))[0]
    h, w = cv2.imread(os.path.join(frames_dir, first)).shape[:2]
    return w, h


def track(frames_dir, text):
    """Run SAM 3.1 on a JPEG folder with one text prompt.

    Returns {frame_idx: [{"id", "p", "box": [x, y, w, h] px, "c": [cx, cy] px, "area"}]}.
    """
    pred = get_predictor()
    W, H = _frame_size(frames_dir)
    sid = pred.handle_request({"type": "start_session", "resource_path": frames_dir})["session_id"]
    pred.handle_request({"type": "add_prompt", "session_id": sid, "frame_index": 0, "text": text})
    rec = {}
    for out in pred.handle_stream_request({"type": "propagate_in_video", "session_id": sid}):
        o = out["outputs"]
        dets = []
        for oid, p, b, m in zip(o["out_obj_ids"], o["out_probs"], o["out_boxes_xywh"], o["out_binary_masks"]):
            m = np.asarray(m).squeeze()
            ys, xs = np.nonzero(m)
            if len(xs):
                sx, sy = W / m.shape[1], H / m.shape[0]
                box = [xs.min() * sx, ys.min() * sy, (xs.max() - xs.min() + 1) * sx, (ys.max() - ys.min() + 1) * sy]
                c = [xs.mean() * sx, ys.mean() * sy]
                area = float(len(xs) * sx * sy)
            else:
                x, y, w, h = [float(v) for v in b]
                if max(x, y, w, h) <= 1.0:
                    x, y, w, h = x * W, y * H, w * W, h * H
                box, c, area = [x, y, w, h], [x + w / 2, y + h / 2], 0.0
            dets.append({"id": int(oid), "p": round(float(p), 3),
                         "box": [round(v, 1) for v in box], "c": [round(v, 1) for v in c], "area": round(area)})
        rec[int(out["frame_index"])] = dets
    pred.handle_request({"type": "close_session", "session_id": sid})
    return rec


def track_clip(video_path, prompts, work_dir, fps_out=10, width=960):
    """Extract frames once, track every prompt. Returns a dict ready to save as JSON."""
    n, fps = extract_frames(video_path, work_dir, fps_out, width)
    W, H = _frame_size(work_dir)
    return {"video": os.path.basename(video_path), "n_frames": n, "fps": fps, "size": [W, H],
            "prompts": {p: track(work_dir, p) for p in prompts}}


def save_tracks(tracks, path):
    with open(path, "w") as f:
        json.dump(tracks, f)


def load_tracks(path):
    with open(path) as f:
        t = json.load(f)
    t["prompts"] = {p: {int(k): v for k, v in rec.items()} for p, rec in t["prompts"].items()}
    return t
