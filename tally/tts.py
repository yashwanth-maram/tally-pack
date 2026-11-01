"""Tally Pack - narration for the demo video (Kokoro-82M, Apache-2.0), run in its own virtualenv.

Usage: python -m tally.tts sentences.json out_dir [voice] [speed]
Writes out_dir/000.wav ... and prints a JSON list of durations in seconds.
"""
import json
import os
import sys

import numpy as np
import soundfile as sf
from kokoro import KPipeline

SR = 24000


def main():
    sents = json.load(open(sys.argv[1]))
    out = sys.argv[2]
    voice = sys.argv[3] if len(sys.argv) > 3 else "am_michael"
    speed = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    os.makedirs(out, exist_ok=True)
    pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
    durs = []
    for i, s in enumerate(sents):
        parts = [np.asarray(a.detach().cpu() if hasattr(a, "detach") else a, dtype=np.float32)
                 for _, _, a in pipe(s, voice=voice, speed=speed) if a is not None]
        audio = np.concatenate(parts) if parts else np.zeros(SR // 2, np.float32)
        sf.write(f"{out}/{i:03d}.wav", audio, SR)
        durs.append(round(len(audio) / SR, 3))
    print(json.dumps(durs))


if __name__ == "__main__":
    main()
