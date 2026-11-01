"""Tally Pack - reasoning backend: Muse Glimmer 30B (Meta open weights, Apache-2.0), run locally.

Loaded in 4-bit (bitsandbytes NF4) so it shares one A100 with SAM 3.1. Every caller asks for a
JSON answer; `ask_json` extracts the last JSON object from the reply (Glimmer is a reasoning
model, so the reply may contain thinking before the answer) and returns None if there is none -
callers must treat None as "no decision" and fall back to the deterministic path.
"""
import json
import re
import time

import torch

MODEL_ID = "meta-models/Muse-Glimmer-30B"
_STATE = {"model": None, "proc": None, "load_s": None}


def load(four_bit=True):
    if _STATE["model"] is not None:
        return _STATE
    from transformers import AutoProcessor, BitsAndBytesConfig
    try:
        from transformers import AutoModelForMultimodalLM as AutoModel
    except ImportError:  # older transformers
        from transformers import AutoModelForImageTextToText as AutoModel
    t0 = time.time()
    qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=torch.bfloat16) if four_bit else None
    _STATE["proc"] = AutoProcessor.from_pretrained(MODEL_ID)
    _STATE["model"] = AutoModel.from_pretrained(MODEL_ID, device_map="auto", dtype=torch.bfloat16,
                                                quantization_config=qc)
    _STATE["model"].eval()
    if four_bit:
        _STATE["dequantized_vision"] = dequantize_vision(_STATE["model"])
    _STATE["load_s"] = round(time.time() - t0, 1)
    return _STATE


VISION_ROOTS = ("vision_tower", "visual", "multi_modal_projector", "mm_projector", "vision_projector")


def dequantize_vision(model):
    """bitsandbytes 4-bit also quantizes Glimmer's vision tower (skip lists are ignored), and its patch
    embedder then casts pixels to uint8. Replace every Linear4bit under the vision modules with an exact
    bf16 nn.Linear (dequantized weights). Returns {module: n_layers_restored}."""
    import bitsandbytes as bnb
    import torch.nn as nn
    restored = {}
    for name, root in [(n, m) for n, m in model.named_modules() if n.split(".")[-1] in VISION_ROOTS]:
        n = 0
        for _, mod in list(root.named_modules()):
            for cname, child in list(mod.named_children()):
                if isinstance(child, bnb.nn.Linear4bit):
                    w = bnb.functional.dequantize_4bit(child.weight.data, child.weight.quant_state).to(torch.bfloat16)
                    lin = nn.Linear(child.in_features, child.out_features, bias=child.bias is not None,
                                    device=w.device, dtype=torch.bfloat16)
                    lin.weight.data = w
                    if child.bias is not None:
                        lin.bias.data = child.bias.data.to(torch.bfloat16)
                    setattr(mod, cname, lin)
                    n += 1
        restored[name] = n
    return restored


FAST_PREFIX = " to=user<|message|>"   # Glimmer's final-answer header; prefilling it skips the reasoning channel


def chat(messages, max_new_tokens=1536, fast=False):
    """messages: [{"role": ..., "content": [{"type": "text", "text": ...} | {"type": "image", "image": PIL}]}]

    fast=True prefills the final-answer header so Glimmer answers directly (seconds instead of minutes).
    """
    st = load()
    proc, model = st["proc"], st["model"]
    text = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    if fast:
        text += FAST_PREFIX
    images = [c["image"] for m in messages for c in m["content"] if c.get("type") == "image"]
    inputs = proc(text=text, images=images or None, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    reply = proc.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return ("to=user" + reply) if fast else reply


def extract_json(text):
    """Glimmer writes its reasoning, then the answer after 'to=user'. Return the largest JSON object
    in the final answer (falling back to the whole text), else None."""
    if not text:
        return None
    for src in (text.split("to=user")[-1], text):
        best = None
        for s0 in [m.start() for m in re.finditer(r"\{", src)]:
            depth = 0
            for i in range(s0, len(src)):
                depth += src[i] == "{"
                depth -= src[i] == "}"
                if depth == 0:
                    try:
                        obj = json.loads(src[s0:i + 1])
                        if best is None or (i + 1 - s0) > best[0]:
                            best = (i + 1 - s0, obj)
                    except json.JSONDecodeError:
                        pass
                    break
        if best:
            return best[1]
    return None


def ask_json(system, user_text, images=(), max_new_tokens=None, fast=True):
    """One-shot JSON question. Returns (parsed_json_or_None, raw_text, seconds).
    fast=True (default): direct answer, ~200 tokens budget. fast=False: full reasoning, 1536 tokens."""
    max_new_tokens = max_new_tokens or (256 if fast else 1536)
    content = [{"type": "image", "image": im} for im in images] + [{"type": "text", "text": user_text}]
    messages = [{"role": "system", "content": [{"type": "text", "text": system}]},
                {"role": "user", "content": content}]
    t0 = time.time()
    raw = chat(messages, max_new_tokens, fast=fast)
    return extract_json(raw), raw, round(time.time() - t0, 1)
