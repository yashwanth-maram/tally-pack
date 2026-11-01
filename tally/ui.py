"""Tally Pack - demo UI (Gradio).

Tab 1 plays the rendered showcase cases (SAM 3.1 masks, live ledger, verdict, Glimmer investigation).
Tab 2 is live: Muse Glimmer compiles a free-text order, then the verifier checks a packed clip against it.
"""
import glob
import json
import os

import gradio as gr

COLOR = {"PASS": "#16a34a", "FAIL": "#dc2626", "NEEDS_REVIEW": "#d97706"}
TITLE = ("# Tally Pack\n**Muse places the order. Tally proves it was packed right.**  \n"
         "SAM 3.1 tracks every item from a fixed camera, a deterministic ledger verifies the box, and "
         "Muse Glimmer investigates only when the evidence is inconsistent. Glimmer can never clear a box: "
         "only the verifier issues PASS or FAIL.")


def badge(verdict, sub=""):
    return (f"<div style='background:{COLOR.get(verdict, '#555')};color:#fff;padding:18px;border-radius:12px;"
            f"font-size:30px;font-weight:800;text-align:center;letter-spacing:1px'>{verdict.replace('_', ' ')}</div>"
            f"<div style='margin-top:8px;font-size:15px'>{sub}</div>")


def _subtitle(s):
    if s["final"] == "PASS":
        return ("Every ordered item is in the box, nothing extra." +
                (" Reached after a verified repair." if s["trace"] else " Zero model calls."))
    if s["final"] == "FAIL":
        return "; ".join(s["final_details"]) or "Box does not match the order."
    return "Evidence stays inconsistent, so Tally refuses to guess and asks a human to sign off."


def _trace_md(s):
    if s["first"] != "NEEDS_REVIEW":
        return "**Investigator:** not needed. The verifier had consistent evidence on the first pass."
    out = ["**Verifier first pass: NEEDS REVIEW**"] + [f"- {v}" for v in s["first_violations"][:4]]
    for t in s["trace"]:
        if t.get("choice") == "escalate":
            out.append(f"\n**Muse Glimmer ({t.get('seconds')}s):** escalate to a human. _{t.get('reason')}_")
        else:
            out.append(f"\n**Muse Glimmer ({t.get('seconds')}s):** {_act(t)} "
                       f"(confidence {t.get('confidence')}). _{t.get('reason')}_")
            nb, na = t.get("integrity_violations") or [0, 0]
            out.append(f"**Verifier re-check:** issues {nb} -> {na}: "
                       f"{'repair ACCEPTED' if t.get('accepted') else 'repair REJECTED'}")
    out.append(f"\n**Final:** {s['final'].replace('_', ' ')} ({s['stop'].replace('_', ' ')})")
    return "\n".join(out)


def _act(t):
    a = t.get("action") or ""
    return a.split(": ", 1)[1] if ": " in a[:4] else a


def _counts(order, counts):
    return [[k, counts.get(k, 0), order.get(k, 0), "ok" if counts.get(k, 0) == order.get(k, 0) else "MISMATCH"]
            for k in sorted(set(order) | set(counts))]


def load_stories(show_dir):
    out = []
    for p in sorted(glob.glob(f"{show_dir}/*.json")):
        s = json.load(open(p))
        if not os.path.exists(s["video"]):            # stories moved to another machine (e.g. the HF Space)
            s["video"] = os.path.join(show_dir, os.path.basename(s["video"]))
        out.append(s)
    return out


def build_ui(show_dir, live=None, intro=""):
    stories = {f"{s['scenario']} | {s['kind']} | {s['name']}": s for s in load_stories(show_dir)}
    labels = list(stories)

    def show(label):
        s = stories[label]
        ev = [[e["t"], e["item"], e["id"], e["event"], e["zone"]] for e in s["events"]]
        return (s["video"], badge(s["final"], _subtitle(s)), _counts(s["order"], s["final_counts"]), _trace_md(s), ev)

    with gr.Blocks(title="Tally Pack") as ui:
        gr.Markdown(TITLE + (("\n\n" + intro) if intro else ""))
        with gr.Tab("Packed boxes"):
            dd = gr.Dropdown(labels, value=labels[0] if labels else None, label="Case (scenario | tracks | clip)")
            with gr.Row():
                vid = gr.Video(label="Fixed camera, SAM 3.1 masks + ledger", autoplay=True, height=430)
                with gr.Column():
                    verdict = gr.HTML()
                    counts = gr.Dataframe(headers=["SKU", "in box", "ordered", "check"], label="Order vs box")
            trace = gr.Markdown()
            events = gr.Dataframe(headers=["t (s)", "track", "ID", "event", "zone"], label="Ledger events")
            outs = [vid, verdict, counts, trace, events]
            dd.change(show, dd, outs)
            ui.load(show, dd, outs)
        if live:
            with gr.Tab(live.get("tab", "Type an order (live Muse Glimmer)")):
                if live.get("note"):
                    gr.Markdown(live["note"])
                with gr.Row():
                    txt = gr.Textbox(label="Order, as the customer wrote it", value="2 maggi, one parle g and a coke", scale=3)
                    clip = gr.Dropdown(live["clips"], value=live["clips"][0], label="Packed box (camera clip)", scale=2)
                btn = gr.Button("Compile the order with Muse Glimmer, then verify the box", variant="primary")
                with gr.Row():
                    contract = gr.JSON(label="Packing contract (Glimmer output, validated)")
                    with gr.Column():
                        v2 = gr.HTML()
                        c2 = gr.Dataframe(headers=["SKU", "in box", "ordered", "check"])
                md2 = gr.Markdown()
                btn.click(live["fn"], [txt, clip], [contract, v2, c2, md2])
                if live.get("examples"):
                    gr.Examples([[t, live["clips"][0]] for t in live["examples"]], [txt, clip], label="Example orders")
    return ui


def make_live(tracks_dir, frames_root, refs, cat_all, clips):
    """Live tab handler: Glimmer compiles the order, the verifier audits the chosen clip against it."""
    from . import agent as A
    from . import contract as C
    from . import llm as LLM
    from . import perception as P

    def fn(text, clip):
        c = C.compile_order(text)
        view = {k: c[k] for k in ("order", "unmatched", "status", "seconds")}
        if c["status"] != "ok":
            return view, badge("NEEDS_REVIEW", "The order has lines Glimmer could not map to the catalogue: "
                               + ", ".join(c["unmatched"]) + ". Tally asks instead of guessing."), [], ""
        t = P.load_tracks(f"{tracks_dir}/{clip}.json")
        fd = f"{frames_root}/{clip}"
        rep = A.audit(t, c["order"], cat_all, fd, refs)
        md = "**Investigator:** not needed (consistent evidence)."
        if rep["verdict"] == "NEEDS_REVIEW":
            r = A.investigate(t, c["order"], cat_all, fd, refs, ask=LLM.ask_json)
            rep = r["report"] | {"verdict": r["verdict"]}
            md = "\n".join(f"- step {x.get('step')}: {_act(x) or 'escalate'} | {x.get('reason')} "
                           f"({x.get('seconds')}s)" for x in r["trace"]) or "- no repair candidates"
        det = "; ".join(v["detail"] for v in rep["violations"]) or "Every ordered item is in the box, nothing extra."
        return view, badge(rep["verdict"], det), _counts(c["order"], rep["counts"]), md

    return {"fn": fn, "clips": clips}


def make_offline_live(data_dir):
    """CPU-only live tab (e.g. a free Hugging Face Space). The verifier runs live on the saved SAM 3.1 tracks
    and saved appearance identities. Orders use the saved Muse Glimmer contract when the text matches an
    example (compiled on the GPU), otherwise the deterministic keyword parser. Nothing is faked: every
    output says where it came from."""
    from . import agent as A
    from . import contract as C
    from . import perception as P

    clips = json.load(open(f"{data_dir}/clips.json"))
    saved = json.load(open(f"{data_dir}/contracts.json"))
    norm = lambda t: " ".join(t.lower().split())
    saved_n = {norm(k): v for k, v in saved.items()}
    cat_all = {}
    for sku, info in C.CATALOG.items():
        cat_all.setdefault(C.CATEGORY_PROMPTS[info["category"]], []).append(sku)

    def fn(text, clip):
        if norm(text) in saved_n:
            c, src = saved_n[norm(text)], f"Muse Glimmer 30B on an A100 (saved output, {saved_n[norm(text)]['seconds']}s)"
        else:
            c, src = C.parse_rules(text), "keyword parser (the 30B model needs a GPU; run the Colab path for live Glimmer)"
        view = {"compiled_by": src, "order": c["order"], "unmatched": c["unmatched"], "status": c["status"]}
        if c["status"] != "ok":
            return view, badge("NEEDS_REVIEW", "These lines did not map to the catalogue: " + ", ".join(c["unmatched"])
                               + ". Tally asks instead of guessing."), [], ""
        t = P.load_tracks(f"{data_dir}/tracks/{clip}.json")
        asg = {p: {int(k): v for k, v in d.items()} for p, d in clips[clip]["assignment"].items()}
        rep = A.audit(t, c["order"], cat_all, None, None, assignment=asg)
        md = ("**Verifier:** ran live here on the saved SAM 3.1 tracks (zero model calls). "
              f"Ground truth for this clip: `{clips[clip]['scenario']}` scenario; the box really holds "
              + ", ".join(f"{v} {k}" for k, v in clips[clip]['in_box'].items()) + ".")
        if rep["verdict"] == "NEEDS_REVIEW":
            md += ("\n\n**Investigator:** the evidence is inconsistent, so this box goes to the Muse Glimmer investigator "
                   "(GPU). See the `occlusion` and `injected` cases in the first tab for real investigation traces.")
        det = "; ".join(v["detail"] for v in rep["violations"]) or "Every ordered item is in the box, nothing extra."
        return view, badge(rep["verdict"], det), _counts(c["order"], rep["counts"]), md

    names = sorted(clips, key=lambda n: (n != "pb_clean_0", n))
    return {"fn": fn, "clips": names, "examples": list(saved), "tab": "Type an order (verifier runs live)",
            "note": "Type an order, pick a packed box, and the verifier checks it **live** on this page. "
                    "The example orders use contracts that Muse Glimmer 30B compiled on the GPU; any other text uses "
                    "a simple keyword parser, because this free CPU Space cannot run a 30B model."}
