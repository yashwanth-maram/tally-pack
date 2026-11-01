"""Tally Pack - packaging: README media, result files, and the CPU-only Hugging Face Space."""
import glob
import json
import os
import shutil
import subprocess

from . import identity as I
from . import perception as P

SPACE_APP = '''import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tally import ui as U  # noqa: E402

INTRO = ("**This page runs on a free CPU.** The showcase videos are real outputs of SAM 3.1 and Muse Glimmer 30B "
         "on an A100. In the second tab the verifier runs live. "
         "[Code](https://github.com/yashwanth-maram/tally-pack) | "
         "[3-minute video](https://github.com/yashwanth-maram/tally-pack/blob/main/docs/tally_pack_demo.mp4)")
demo = U.build_ui(f"{HERE}/data/showcase", live=U.make_offline_live(f"{HERE}/data"), intro=INTRO)
demo.launch(allowed_paths=[f"{HERE}/data"])
'''

SPACE_README = '''---
title: Tally Pack
emoji: 📦
colorFrom: yellow
colorTo: gray
sdk: gradio
sdk_version: {gradio}
app_file: app.py
pinned: true
license: mit
short_description: Camera-verified order packing with SAM 3.1 + Muse Glimmer
---

# Tally Pack: Muse places the order. Tally proves it was packed right.

Live demo of [Tally Pack](https://github.com/yashwanth-maram/tally-pack), built by Maram Yashwanth for the
Meta Global AI Developer Hackathon. SAM 3.1 tracks every item from a fixed camera, a deterministic ledger
verifies the box, and Muse Glimmer investigates only when the evidence is inconsistent.
'''


def gif(src, dst, start=0.0, dur=None, width=760, fps=8, speed=1.0):
    """High-quality GIF from an mp4 (two-pass palette)."""
    t = ["-ss", str(start)] + (["-t", str(dur)] if dur else [])
    vf = f"setpts=PTS/{speed},fps={fps},scale={width}:-1:flags=lanczos"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *t, "-i", src, "-vf",
                    f"{vf},split[a][b];[a]palettegen=max_colors=160:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4",
                    dst], check=True)
    return os.path.getsize(dst)


def still(src, dst, at=-0.5, width=1280):
    """One frame (default: 0.5 s before the end, where the verdict is on screen)."""
    args = ["-sseof", str(at)] if at < 0 else ["-ss", str(at)]
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args, "-i", src, "-frames:v", "1",
                    "-vf", f"scale={width}:-1", "-q:v", "3", dst], check=True)
    return dst


def readme_media(show_dir, video, out_dir):
    """docs/: the submission video, GIFs and verdict stills referenced by README.md."""
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy(video, f"{out_dir}/tally_pack_demo.mp4")
    sizes = {
        "hero.gif": gif(f"{show_dir}/pb_clean_0__injected.mp4", f"{out_dir}/hero.gif", width=820, fps=8, speed=1.4),
        "missing.gif": gif(f"{show_dir}/pb_missing_0__normal.mp4", f"{out_dir}/missing.gif", width=560, fps=6, speed=1.6),
        "occlusion.gif": gif(f"{show_dir}/pb_occlusion_4__normal.mp4", f"{out_dir}/occlusion.gif", width=560, fps=6, speed=1.4),
    }
    for k in ("pb_clean_0__normal", "pb_missing_0__normal", "pb_extra_0__normal", "pb_wrong_0__normal",
              "pb_twins_0__normal", "pb_occlusion_4__normal", "pb_clean_0__injected", "pb_missing_0__injected"):
        if os.path.exists(f"{show_dir}/{k}.mp4"):
            still(f"{show_dir}/{k}.mp4", f"{out_dir}/{k}.jpg", width=900)
    return sizes


def results_files(out_dir, rows, gts, extra=None):
    """results/: the agent evaluation, the benchmark ground truth (without per-frame boxes) and a summary."""
    os.makedirs(out_dir, exist_ok=True)
    json.dump(rows, open(f"{out_dir}/agent_eval_glimmer_fast.json", "w"), indent=1, default=str)
    slim = [{k: v for k, v in g.items() if k not in ("truth", "events")} for g in gts]
    json.dump(slim, open(f"{out_dir}/packbench_ground_truth.json", "w"), indent=1)
    if extra:
        json.dump(extra, open(f"{out_dir}/summary.json", "w"), indent=1)


def build_space(space_dir, repo_dir, show_dir, gts, tracks_dir, frames_root, refs, cat_all, contracts, gradio_version):
    """Self-contained CPU Space: code + showcase videos + tracks + precomputed appearance identities."""
    shutil.rmtree(space_dir, ignore_errors=True)
    os.makedirs(f"{space_dir}/data/tracks")
    shutil.copytree(f"{repo_dir}/tally", f"{space_dir}/tally", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(show_dir, f"{space_dir}/data/showcase", ignore=shutil.ignore_patterns("masks", "*.pkl.gz"))
    clips = {}
    for g in gts:
        name = g["name"]
        t = P.load_tracks(f"{tracks_dir}/{name}.json")
        P.save_tracks(t, f"{space_dir}/data/tracks/{name}.json")
        cats = {p: s for p, s in cat_all.items() if p in t["prompts"]}
        asg = I.assign(f"{frames_root}/{name}", t, cats, refs)
        clips[name] = {"scenario": g["scenario"], "order": g["order"], "in_box": g["in_box"],
                       "expected": g["expected"], "assignment": asg}
    json.dump(clips, open(f"{space_dir}/data/clips.json", "w"), default=float)
    json.dump(contracts, open(f"{space_dir}/data/contracts.json", "w"), indent=1)
    open(f"{space_dir}/app.py", "w").write(SPACE_APP)
    open(f"{space_dir}/requirements.txt", "w").write("numpy\nopencv-python-headless\nPillow\n")
    open(f"{space_dir}/README.md", "w").write(SPACE_README.format(gradio=gradio_version))
    mb = sum(os.path.getsize(p) for p in glob.glob(f"{space_dir}/**/*", recursive=True) if os.path.isfile(p)) / 1e6
    return {"clips": len(clips), "contracts": len(contracts), "size_mb": round(mb, 1)}
