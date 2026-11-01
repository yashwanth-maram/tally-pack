"""Tally Pack - demo UI (Gradio).

Tab 1 plays the rendered showcase cases (SAM 3.1 masks, live ledger, verdict, Glimmer investigation).
Tab 2 is live: Muse Glimmer compiles a free-text order, then the verifier checks a packed clip against it.
"""
import glob
import json

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
    return [json.load(open(p)) for p in sorted(glob.glob(f"{show_dir}/*.json"))]


def build_ui(show_dir, live=None):
    stories = {f"{s['scenario']} | {s['kind']} | {s['name']}": s for s in load_stories(show_dir)}
    labels = list(stories)

    def show(label):
        s = stories[label]
        ev = [[e["t"], e["item"], e["id"], e["event"], e["zone"]] for e in s["events"]]
        return (s["video"], badge(s["final"], _subtitle(s)), _counts(s["order"], s["final_counts"]), _trace_md(s), ev)

    with gr.Blocks(title="Tally Pack") as ui:
        gr.Markdown(TITLE)
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
            with gr.Tab("Type an order (live Muse Glimmer)"):
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
