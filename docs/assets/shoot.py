"""Re-render the README art cards: python docs/assets/shoot.py (needs playwright + chromium)."""
import os
from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1700, "height": 1200}, device_scale_factor=1.25)
    pg.goto(f"file://{HERE}/art.html")
    pg.wait_for_timeout(800)
    for el, name in [("banner", "banner"), ("arch", "architecture"), ("score", "results"), ("verd", "verdicts")]:
        pg.locator("#" + el).screenshot(path=f"{HERE}/{name}.png", omit_background=True)
    b.close()
