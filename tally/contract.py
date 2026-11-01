"""Tally Pack - contract compiler: free-text order -> machine-checkable packing contract.

Muse Glimmer reads the order the way a person would write it ("2 maggi masala + a coke pls")
and maps it onto the store catalogue. Its output is never trusted blindly: every line is
validated deterministically (SKU must exist, quantity must be a sane integer); anything that
fails validation is returned as `unmatched` so the packer is asked instead of guessed.
"""
import json
import re

CATALOG = {
    "maggi": {"name": "Maggi 2-Minute Masala Noodles (packet)", "category": "packet"},
    "parle_g": {"name": "Parle-G Glucose Biscuits (packet)", "category": "packet"},
    "lays": {"name": "Lay's potato chips (bag)", "category": "packet"},
    "coke_can": {"name": "Coca-Cola (can)", "category": "can"},
    "dettol": {"name": "Dettol (plastic bottle)", "category": "bottle"},
}
CATEGORY_PROMPTS = {"packet": "packet", "can": "soda can", "bottle": "plastic bottle"}

SYSTEM = ("You convert a customer's packing order into JSON for a warehouse verifier. "
          "Map every requested product to exactly one catalogue SKU id, or list it as unmatched if no SKU fits. "
          "Never invent SKUs. Answer with only this JSON object: "
          '{"items": [{"sku": "<id>", "qty": <int>}], "unmatched": ["<text>"]}')


def _prompt(order_text, catalog):
    cat = "\n".join(f"- {k}: {v['name']}" for k, v in catalog.items())
    return f"Catalogue:\n{cat}\n\nOrder: {order_text}\n\nReturn the JSON object."


def validate(parsed, catalog, max_qty=20):
    items, unmatched = {}, []
    if not isinstance(parsed, dict):
        return items, ["<model returned no JSON>"]
    for it in parsed.get("items", []) or []:
        sku, qty = (it or {}).get("sku"), (it or {}).get("qty")
        if sku in catalog and isinstance(qty, int) and 1 <= qty <= max_qty:
            items[sku] = items.get(sku, 0) + qty
        else:
            unmatched.append(json.dumps(it))
    unmatched += [str(u) for u in parsed.get("unmatched", []) or []]
    return items, unmatched


def compile_order(order_text, catalog=CATALOG):
    """Returns the contract: order counts, SAM prompts per category, and any unmatched lines."""
    from . import llm  # lazy: the verifier and the CPU demo never need torch
    parsed, raw, secs = llm.ask_json(SYSTEM, _prompt(order_text, catalog))
    return _contract(order_text, *validate(parsed, catalog), catalog, llm.MODEL_ID, secs)


def _contract(order_text, items, unmatched, catalog, model, secs):
    cats = {}
    for sku in items:
        cats.setdefault(CATEGORY_PROMPTS[catalog[sku]["category"]], []).append(sku)
    return {"order_text": order_text, "order": items, "category_prompts": cats, "unmatched": unmatched,
            "status": "ok" if items and not unmatched else ("needs_input" if unmatched else "empty"),
            "model": model, "seconds": secs}


# ------------------------------------------------------------------ keyword fallback (no model)
ALIASES = {"maggi": ("maggi", "noodle"), "parle_g": ("parle", "biscuit"), "lays": ("lay", "chip", "crisp"),
           "coke_can": ("coke", "coca", "cola", "soda"), "dettol": ("dettol", "sanitiz", "sanitis", "handwash")}
NUMBER_WORDS = {"a": 1, "an": 1, "one": 1, "single": 1, "two": 2, "pair": 2, "couple": 2, "three": 3,
                "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def parse_rules(order_text, catalog=CATALOG):
    """Deterministic keyword parser, used where the 30B model cannot run (e.g. a free CPU demo).
    Each comma / 'and' / '+' separated part must name exactly one catalogue product."""
    items, unmatched = {}, []
    for part in re.split(r",|;|\+|&|\band\b|\n", order_text.lower()):
        toks = re.findall(r"[a-z]+|\d+", part)
        if not toks:
            continue
        skus = [s for s, keys in ALIASES.items() if s in catalog and any(t.startswith(k) for t in toks for k in keys)]
        nums = [int(t) for t in toks if t.isdigit()] + [NUMBER_WORDS[t] for t in toks if t in NUMBER_WORDS]
        if len(skus) == 1 and (nums[0] if nums else 1) <= 20:
            items[skus[0]] = items.get(skus[0], 0) + (nums[0] if nums else 1)
        elif any(t not in NUMBER_WORDS and not t.isdigit() and t not in ("pls", "please", "x", "of", "packet", "pack", "can", "bottle", "bottles", "packets") for t in toks):
            unmatched.append(part.strip())
    return _contract(order_text, items, unmatched, catalog, "keyword parser", 0.0)
