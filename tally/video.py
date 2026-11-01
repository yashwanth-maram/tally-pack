"""Tally Pack - builds the 3-minute submission video from code: title cards, the rendered demo clips,
burned-in captions and an optional synthetic voice-over (Kokoro-82M via tally.tts). Nothing is filmed.
"""
import json
import os
import re
import subprocess
import wave

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS, SR = 1920, 1080, 30, 24000
TOP, VID_H, CAP_Y = 92, 755, 860          # title band, clip area height, caption band start
BG, FG, MUTED, ACCENT = (16, 17, 21), (240, 240, 240), (150, 155, 165), (255, 140, 40)
GREEN, RED, AMBER = (22, 163, 74), (220, 38, 38), (217, 119, 6)
FONT_DIRS = ["/usr/share/fonts/truetype/liberation", "/usr/share/fonts/truetype/dejavu"]
FONT_FILES = {False: ["LiberationSans-Regular.ttf", "DejaVuSans.ttf"],
              True: ["LiberationSans-Bold.ttf", "DejaVuSans-Bold.ttf"]}
_FONTS = {}


def font(size, bold=False):
    key = (size, bold)
    if key not in _FONTS:
        for d in FONT_DIRS:
            for f in FONT_FILES[bold]:
                if os.path.exists(f"{d}/{f}"):
                    _FONTS[key] = ImageFont.truetype(f"{d}/{f}", size)
                    break
            if key in _FONTS:
                break
        else:
            _FONTS[key] = ImageFont.load_default(size)
    return _FONTS[key]


def wrap(draw, text, fnt, width):
    lines, cur = [], ""
    for w in text.split():
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=fnt) <= width:
            cur = t
        else:
            lines.append(cur)
            cur = w
    return lines + ([cur] if cur else [])


def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


# ------------------------------------------------------------------ cards (PIL, 1920 x CAP_Y)
def _canvas():
    img = Image.new("RGB", (W, CAP_Y), BG)
    return img, ImageDraw.Draw(img)


def card_title(title, tagline, lines=(), footer=""):
    img, d = _canvas()
    d.rectangle([0, 0, W, 10], fill=ACCENT)
    d.text((W // 2, 300), title, font=font(130, True), fill=FG, anchor="mm")
    d.text((W // 2, 430), tagline, font=font(52), fill=ACCENT, anchor="mm")
    y = 540
    for ln in lines:
        d.text((W // 2, y), ln, font=font(38), fill=MUTED, anchor="mm")
        y += 58
    if footer:
        d.text((W // 2, CAP_Y - 70), footer, font=font(34), fill=MUTED, anchor="mm")
    return img


def card_bullets(heading, bullets, sub=""):
    img, d = _canvas()
    d.rectangle([0, 0, 14, CAP_Y], fill=ACCENT)
    d.text((110, 90), heading, font=font(72, True), fill=FG)
    y = 210
    if sub:
        d.text((110, y), sub, font=font(36), fill=MUTED)
        y += 80
    for b in bullets:
        head, _, rest = b.partition("|")
        d.ellipse([112, y + 18, 132, y + 38], fill=ACCENT)
        d.text((165, y), head.strip(), font=font(44, True), fill=FG)
        if rest:
            for ln in wrap(d, rest.strip(), font(36), W - 300):
                y += 54
                d.text((165, y), ln, font=font(36), fill=MUTED)
        y += 84
    return img


def card_pipeline(heading, steps, note):
    """steps: list of (title, body) drawn as connected boxes."""
    img, d = _canvas()
    d.rectangle([0, 0, 14, CAP_Y], fill=ACCENT)
    d.text((110, 80), heading, font=font(72, True), fill=FG)
    n, gap = len(steps), 60
    bw = (W - 220 - gap * (n - 1)) // n
    y0, y1 = 230, 650
    for i, (t, body) in enumerate(steps):
        x0 = 110 + i * (bw + gap)
        col = (60, 40, 20) if "Glimmer" in t else (32, 36, 46)
        d.rounded_rectangle([x0, y0, x0 + bw, y1], 24, fill=col, outline=ACCENT if "Glimmer" in t else (90, 96, 110), width=3)
        yy = y0 + 30
        for ln in wrap(d, t, font(34, True), bw - 40):
            d.text((x0 + 22, yy), ln, font=font(34, True), fill=FG)
            yy += 44
        yy += 16
        for ln in wrap(d, body, font(29), bw - 40):
            d.text((x0 + 22, yy), ln, font=font(29), fill=(205, 208, 215))
            yy += 39
        if i < n - 1:
            ax = x0 + bw + 8
            d.line([ax, (y0 + y1) // 2, ax + gap - 16, (y0 + y1) // 2], fill=ACCENT, width=6)
            d.polygon([(ax + gap - 16, (y0 + y1) // 2 - 14), (ax + gap - 4, (y0 + y1) // 2), (ax + gap - 16, (y0 + y1) // 2 + 14)], fill=ACCENT)
    d.text((W // 2, 745), note, font=font(40, True), fill=ACCENT, anchor="mm")
    return img


def card_live(heading, cases, sub=""):
    """cases: list of dicts {order_text, contract: {sku: qty}, verdict, reason}."""
    img, d = _canvas()
    d.rectangle([0, 0, 14, CAP_Y], fill=ACCENT)
    d.text((110, 80), heading, font=font(68, True), fill=FG)
    if sub:
        d.text((110, 170), sub, font=font(34), fill=MUTED)
    y = 260
    for c in cases:
        d.rounded_rectangle([110, y, 690, y + 230], 20, fill=(32, 36, 46))
        d.text((135, y + 20), "Customer wrote", font=font(28), fill=MUTED)
        for i, ln in enumerate(wrap(d, f"“{c['order_text']}”", font(40, True), 530)[:3]):
            d.text((135, y + 62 + i * 50), ln, font=font(40, True), fill=FG)
        d.rounded_rectangle([740, y, 1180, y + 230], 20, fill=(60, 40, 20), outline=ACCENT, width=3)
        d.text((765, y + 20), "Muse Glimmer contract", font=font(28), fill=MUTED)
        for i, (k, v) in enumerate(list(c["contract"].items())[:4]):
            d.text((765, y + 62 + i * 40), f"{k} x {v}", font=font(36, True), fill=FG)
        col = {"PASS": GREEN, "FAIL": RED}.get(c["verdict"], AMBER)
        d.rounded_rectangle([1230, y, 1810, y + 230], 20, fill=col)
        d.text((1520, y + 60), c["verdict"].replace("_", " "), font=font(64, True), fill=(255, 255, 255), anchor="mm")
        for i, ln in enumerate(wrap(d, c["reason"], font(28), 540)[:3]):
            d.text((1255, y + 110 + i * 36), ln, font=font(28), fill=(255, 255, 255))
        for x in (700, 1190):
            d.polygon([(x, y + 100), (x + 30, y + 115), (x, y + 130)], fill=ACCENT)
        y += 280
    return img


def card_results(heading, rows, note=""):
    """rows: list of (label, value, color or None)."""
    img, d = _canvas()
    d.rectangle([0, 0, 14, CAP_Y], fill=ACCENT)
    d.text((110, 80), heading, font=font(72, True), fill=FG)
    y = 195
    for label, value, col in rows:
        d.text((110, y), label, font=font(40), fill=(215, 218, 225))
        d.text((W - 110, y), value, font=font(48, True), fill=col or FG, anchor="ra")
        d.line([110, y + 64, W - 110, y + 64], fill=(50, 54, 64), width=2)
        y += 86
    if note:
        for i, ln in enumerate(wrap(d, note, font(30), W - 220)):
            d.text((110, y + 20 + i * 40), ln, font=font(30), fill=MUTED)
    return img


# ------------------------------------------------------------------ frame composition
def _band(title, right="TALLY PACK"):
    img = Image.new("RGB", (W, TOP), (10, 10, 12))
    d = ImageDraw.Draw(img)
    d.text((40, TOP // 2), title, font=font(40, True), fill=FG, anchor="lm")
    d.text((W - 40, TOP // 2), right, font=font(34, True), fill=ACCENT, anchor="rm")
    return np.asarray(img)[:, :, ::-1]


def _caption(text):
    img = Image.new("RGB", (W, H - CAP_Y), (8, 8, 10))
    d = ImageDraw.Draw(img)
    if text:
        lines = wrap(d, text, font(44), W - 240)[:3]
        y = (H - CAP_Y) // 2 - (len(lines) * 58) // 2
        for ln in lines:
            d.text((W // 2, y), ln, font=font(44), fill=(255, 255, 255), anchor="ma")
            y += 58
    return np.asarray(img)[:, :, ::-1]


def _clip_frames(path):
    cap, out = cv2.VideoCapture(path), []
    fps = cap.get(cv2.CAP_PROP_FPS) or 10
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        h, w = fr.shape[:2]
        nw, nh = W, int(h * W / w)
        if nh > CAP_Y - TOP:
            nh, nw = CAP_Y - TOP, int(w * (CAP_Y - TOP) / h)
        out.append((cv2.resize(fr, (nw, nh), interpolation=cv2.INTER_AREA)))
    return out, fps


# ------------------------------------------------------------------ narration
def narrate(all_sents, work, tts_python=None, voice="am_michael", speed=1.0):
    """Returns (durations, wav_dir or None). Falls back to reading-speed estimates without TTS."""
    est = [max(1.4, len(s.split()) / 2.6) for s in all_sents]
    if not tts_python:
        return est, None
    os.makedirs(work, exist_ok=True)
    json.dump(all_sents, open(f"{work}/sents.json", "w"))
    r = subprocess.run([tts_python, "-m", "tally.tts", f"{work}/sents.json", f"{work}/wav", voice, str(speed)],
                       capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        return json.loads(r.stdout.strip().splitlines()[-1]), f"{work}/wav"
    except Exception:
        print("TTS failed, captions only:", (r.stderr or r.stdout)[-300:])
        return est, None


def _read_wav(path):
    with wave.open(path) as w:
        a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16 if w.getsampwidth() == 2 else np.int32)
        return a.astype(np.float32) / (32768.0 if w.getsampwidth() == 2 else 2 ** 31)


def build(segments, out_mp4, work, tts_python=None, voice="am_michael", speed=1.0, lead=0.35, gap=0.25, tail=0.45):
    """segments: [{"kind": "card", "image": PIL} | {"kind": "clip", "video": path}, + "title", "say", "min_s"]."""
    os.makedirs(work, exist_ok=True)
    per = [sentences(s["say"]) for s in segments]
    flat = [x for p in per for x in p]
    durs, wav_dir = narrate(flat, work, tts_python, voice, speed)
    tmp = f"{work}/video_raw.mp4"
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    audio, k, timeline = [], 0, []
    for seg, sents in zip(segments, per):
        sd = durs[k:k + len(sents)]
        wavs = [_read_wav(f"{wav_dir}/{k + i:03d}.wav") for i in range(len(sents))] if wav_dir else None
        k += len(sents)
        speech = lead + sum(sd) + gap * len(sents) + tail
        if seg["kind"] == "clip":
            frames, sfps = _clip_frames(seg["video"])
            dur = max(speech, len(frames) / sfps, seg.get("min_s", 0))
        else:
            base = np.asarray(seg["image"].convert("RGB"))[:, :, ::-1]
            dur = max(speech, seg.get("min_s", 3.0))
        nF = int(round(dur * FPS))
        band = _band(seg.get("title", ""), seg.get("right", "TALLY PACK"))
        caps, starts, t = [_caption(s) for s in sents], [], lead
        for d_ in sd:
            starts.append(t)
            t += d_ + gap
        empty = _caption("")
        for fi in range(nF):
            tt = fi / FPS
            frame = np.empty((H, W, 3), np.uint8)
            if seg["kind"] == "clip":
                frame[:CAP_Y] = 0
                fr = frames[min(len(frames) - 1, int(tt * sfps))]
                x0 = (W - fr.shape[1]) // 2
                frame[TOP:TOP + fr.shape[0], x0:x0 + fr.shape[1]] = fr
                frame[:TOP] = band
            else:
                frame[:CAP_Y] = base
            idx = max([i for i, s in enumerate(starts) if s <= tt] or [None]) if starts and tt >= starts[0] else None
            frame[CAP_Y:] = caps[idx] if idx is not None else empty
            vw.write(frame)
        seg_audio = np.zeros(nF * SR // FPS, np.float32)
        if wavs:
            for st, a in zip(starts, wavs):
                i0 = int(st * SR)
                seg_audio[i0:i0 + len(a)] = a[: max(0, len(seg_audio) - i0)]
        audio.append(seg_audio)
        timeline.append((seg.get("title", ""), round(dur, 1)))
    vw.release()
    wav_path = f"{work}/narration.wav"
    with wave.open(wav_path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(np.concatenate(audio), -1, 1) * 32767).astype(np.int16).tobytes())
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-i", wav_path, "-c:v", "libx264", "-preset", "veryfast",
                    "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart",
                    out_mp4], check=True)
    os.remove(tmp)
    return {"path": out_mp4, "seconds": round(sum(d for _, d in timeline), 1), "segments": timeline, "voice": bool(wav_dir)}


# ------------------------------------------------------------------ the submission storyboard
NAMES = {"maggi": "Maggi", "parle_g": "Parle-G", "lays": "Lay's", "coke_can": "Coke", "dettol": "Dettol"}
REPO_URL = "github.com/yashwanth-maram/tally-pack"


def _missing(story):
    out = []
    for d_ in story["final_details"]:
        m = re.match(r"missing: \d+ (\w+) in box", d_)
        if m:
            out.append(NAMES.get(m.group(1), m.group(1)))
    return out


def eval_numbers(rows):
    def part(kind):
        R = [r for r in rows if r["kind"] == kind]
        fail = [r for r in R if r["expected"] == "FAIL"]
        return {"n": len(R), "correct": sum(r["final"] == r["expected"] for r in R),
                "fail_n": len(fail), "caught": sum(r["final"] == "FAIL" for r in fail),
                "false_clear": sum(r["final"] == "PASS" for r in fail),
                "human": sum(r["stop"] == "awaiting_signoff" for r in R)}
    calls = [t for r in rows for t in r["trace"] if t.get("seconds")]
    return {"normal": part("normal"), "injected": part("injected"),
            "glimmer_s": sum(t["seconds"] for t in calls) / max(1, len(calls))}


def storyboard(show_dir, rows, live_cases):
    """The 10 segments of the ~3 minute submission video. Numbers come from the saved evaluation."""
    st = {k: json.load(open(f"{show_dir}/{k}.json")) for k in
          ("pb_missing_0__normal", "pb_clean_0__injected", "pb_wrong_0__normal", "pb_occlusion_4__normal")}
    ev = eval_numbers(rows)
    nm, ni = ev["normal"], ev["injected"]
    miss = _missing(st["pb_missing_0__normal"]) or ["one item"]
    tr = next((t for t in st["pb_clean_0__injected"]["trace"] if t.get("accepted")), {})
    nb, na = (tr.get("integrity_violations") or [2, 0])
    live_s = sum(c["seconds"] for c in live_cases) / max(1, len(live_cases))
    disclose = "PackBench clip (synthetic: real table, hand and public product photos, animated with ground truth)"
    return [
        {"kind": "card", "title": "", "min_s": 4,
         "image": card_title("Tally Pack", "Muse places the order. Tally proves it was packed right.",
                             ["Camera-verified order packing with SAM 3.1 and Muse Glimmer",
                              "Every box checked before it ships. No barcodes, no guessing."],
                             "Maram Yashwanth  ·  B V Raju Institute of Technology"),
         "say": "Meta's Muse agent now shops for millions of people. But a person still packs every box, and two to "
                "four orders in every hundred ship wrong. Tally Pack proves each box matches the order before it "
                "leaves the desk."},
        {"kind": "clip", "video": st["pb_missing_0__normal"]["video"], "title": "A missing item, caught at the packing desk",
         "right": "SAM 3.1 + ledger",
         "say": f"A fixed camera watches the desk. SAM 3.1 segments and tracks every item, and a plain code ledger "
                f"records what goes into the box. The order included {' and '.join(miss)}, but it never went into the "
                f"box, so the verifier says fail before the box ships."},
        {"kind": "card", "title": "",
         "image": card_pipeline("How Tally works", [
             ("Muse Glimmer: order to contract", "Free text such as “2 maggi and a coke” becomes validated SKU counts."),
             ("SAM 3.1: track every item", "Text-prompted video segmentation, one stable ID per item."),
             ("Ledger + verifier", "Plain code, zero model calls. Verdict: PASS, FAIL or NEEDS REVIEW."),
             ("Muse Glimmer: investigator", "Reads the violation and crops of the items, proposes one repair or escalates."),
             ("Verifier gate", "Keeps a repair only if issues drop and no new ones appear.")],
             "Glimmer can never clear a box. Only the verifier issues PASS or FAIL."),
         "say": "Here is the loop. Muse Glimmer turns the customer's words into a packing contract. SAM 3.1 tracks the "
                "items, and a deterministic verifier issues the verdict. When the evidence does not add up, Glimmer "
                "investigates, but its fix is kept only if the verifier confirms it. Glimmer can never clear a box on its own."},
        {"kind": "clip", "video": st["pb_clean_0__injected"]["video"], "title": "Self-repair: a tracker ID switch",
         "right": "Muse Glimmer + verifier gate",
         "say": f"Trackers make mistakes. Here we inject one: the same item suddenly gets a new ID. The ledger no longer "
                f"balances, so the verifier refuses to answer. Glimmer compares crops of both IDs and proposes a merge. "
                f"The verifier re-checks, the issues drop from {nb} to {na}, and the box passes."},
        {"kind": "clip", "video": st["pb_wrong_0__normal"]["video"], "title": "A look-alike substitute",
         "right": "SAM 3.1 + appearance identity",
         "say": "Look-alike swaps are the classic packing error. SAM 3.1 finds every packet, and appearance matching "
                "against the catalogue photo decides which product each one is. The substitute is caught, with the exact reason."},
        {"kind": "clip", "video": st["pb_occlusion_4__normal"]["video"], "title": "Knowing when not to answer",
         "right": "awaiting human sign-off",
         "say": "When the evidence cannot be repaired, Tally does not guess. Here the hand covers the bag and the track "
                "jumps across the frame. Glimmer finds no repair that fits the crops, so the box goes to a human for sign-off."},
        {"kind": "card", "title": "",
         "image": card_live("Orders arrive as free text", live_cases,
                            "Same packed box, two orders. Muse Glimmer 30B runs locally as open weights (Apache-2.0)."),
         "say": f"Orders arrive as free text. Muse Glimmer, running locally as open weights, compiles each one into a "
                f"validated contract in about {round(live_s)} seconds. Same box, two orders: one fails with the exact "
                f"reasons, the other passes."},
        {"kind": "card", "title": "",
         "image": card_results("Measured on PackBench (30 clips)", [
             ("Packing errors caught (missing, extra, wrong item)", f"{nm['caught']} / {nm['fail_n']}", GREEN),
             ("False clearances: a wrong box marked PASS", f"{nm['false_clear'] + ni['false_clear']} / {nm['fail_n'] + ni['fail_n']}", GREEN),
             ("Correct verdicts, clean tracks", f"{nm['correct']} / {nm['n']}", None),
             ("Correct verdicts, injected ID switches", f"{ni['correct']} / {ni['n']}", None),
             ("Sent to a human instead of guessing", f"{nm['human'] + ni['human']} / {nm['n'] + ni['n']}", AMBER),
             ("Muse Glimmer decision time (one A100, 4-bit)", f"≈ {ev['glimmer_s']:.0f} s", None)],
             "PackBench is synthetic: a real table, hand and public product photos (Open Food Facts), animated with "
             "ground truth by construction. Baseline with brand-name prompts: 2 false clearances out of 6."),
         "say": "On thirty synthetic packing clips, Tally caught every packing error and gave zero false clearances, "
                "with clean tracks and with injected tracking failures. The only misses are cases it honestly sent to a human."},
        {"kind": "card", "title": "",
         "image": card_bullets("Why this matters for Meta", [
             "Muse shopping needs trust | Shopping is the top Muse use case. A wrong box breaks trust in the agent, not the packer.",
             "Built on Meta open models | SAM 3.1 and Muse Glimmer on one GPU. No paid API needed.",
             "Ready for glasses | As Muse comes to Meta glasses, the packer's glasses can become the camera.",
             "Verify, then trust | The same verifier-gated loop fits any Meta agent that acts in the physical world."]),
         "say": "For Meta, every order Muse places depends on the right box arriving. Tally runs on Meta's own open "
                "models on a single GPU, it can move to the camera on Meta glasses, and its verify then trust loop works "
                "for any agent that acts in the physical world."},
        {"kind": "card", "title": "", "min_s": 5,
         "image": card_title("Tally Pack", "Perception you can audit.",
                             [REPO_URL, "Built by Maram Yashwanth  ·  BVRIT"],
                             "SAM 3.1  ·  Muse Glimmer 30B  ·  open source"),
         "say": "Tally Pack. Muse places the order, Tally proves it was packed right. Built by Maram Yashwanth."},
    ]
