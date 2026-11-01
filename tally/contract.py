"""Tally Pack - contract compiler: free-text order -> machine-checkable packing contract.

Muse Glimmer reads the order the way a person would write it ("2 maggi masala + a coke pls")
and maps it onto the store catalogue. Its output is never trusted blindly: every line is
validated deterministically (SKU must exist, quantity must be a sane integer); anything that
fails validation is returned as `unmatched` so the packer is asked instead of guessed.
"""
import json

from . import llm

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
    parsed, raw, secs = llm.ask_json(SYSTEM, _prompt(order_text, catalog), max_new_tokens=2048)
    items, unmatched = validate(parsed, catalog)
    cats = {}
    for sku in items:
        cats.setdefault(CATEGORY_PROMPTS[catalog[sku]["category"]], []).append(sku)
    return {"order_text": order_text, "order": items, "category_prompts": cats, "unmatched": unmatched,
            "status": "ok" if items and not unmatched else ("needs_input" if unmatched else "empty"),
            "model": llm.MODEL_ID, "seconds": secs}
