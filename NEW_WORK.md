# New work statement

All code in this repository was written during the **Meta Global AI Developer Hackathon**
(Oct 26 - Nov 4, 2026). Development started Oct 31 / Nov 1, 2026 (IST); the commit history is the record.

**Author:** Maram Yashwanth (B V Raju Institute of Technology), 24211a66a7@bvrit.ac.in

## What was built during the event
- `tally/perception.py`: SAM 3.1 tracking wrapper for Colab (A100, Python 3.13).
- `tally/ledger.py`: ledger, invariants C2-C4 and the verified repair gate.
- `tally/identity.py`: appearance identity and the C1/C5 checks.
- `tally/assets.py`, `tally/packbench.py`, `tally/evaluate.py`: the PackBench benchmark and its evaluation.
- `tally/llm.py`, `tally/contract.py`, `tally/agent.py`: Muse Glimmer 30B (4-bit + bf16 vision, fast mode), the contract compiler and the investigator agent.
- `tally/demo.py`, `tally/ui.py`, `tally/video.py`, `tally/tts.py`, `tally/publish.py`: demo videos, Gradio app, submission video, Hugging Face Space.
- `results/`, `docs/`: evaluation outputs, README media and the 3-minute demo video.

## Not new
- Meta's SAM 3.1 and Muse Glimmer 30B (used as models), Kokoro-82M (narration voice), and public
  Pexels / Open Food Facts / Wikimedia Commons media (credited in README.md).

## Tools
- Development was assisted by Claude (Anthropic) as a coding assistant; those commits carry a `Co-Authored-By` trailer.
