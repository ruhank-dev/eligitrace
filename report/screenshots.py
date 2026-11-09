"""Capture real screenshots of the running app (task app) for the report/README.

uv run --with playwright python report/screenshots.py   (first time: uv run --with playwright playwright install chromium)
"""

import json
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:8000"
OUT = Path(__file__).resolve().parent / "figures" / "screens"
OUT.mkdir(parents=True, exist_ok=True)

UNSEEN_NOTE = (
    "A 67-year-old man with type 2 diabetes mellitus diagnosed 12 years ago presents for routine follow-up. "
    "His most recent HbA1c was 8.4% despite metformin 1000 mg twice daily. He has stage 3 chronic kidney disease "
    "with an eGFR of 48 mL/min/1.73m2. He had a myocardial infarction four years ago treated with a stent. "
    "He quit smoking 20 years ago. He is not on insulin. He lives alone and is able to give informed consent."
)
UNSEEN_CRITERIA = (
    "Inclusion Criteria:\n- Adults aged 40 to 80 years with type 2 diabetes\n- HbA1c between 7.0% and 10.5%\n"
    "- Stable dose of metformin for at least 3 months\nExclusion Criteria:\n- Current insulin therapy\n"
    "- eGFR below 30 mL/min/1.73m2\n- Pregnancy or breastfeeding\n- History of pancreatitis"
)


def wait_ready(timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            h = json.loads(urllib.request.urlopen(f"{BASE}/api/health", timeout=5).read())
            if h["status"] == "ready":
                return h
        except Exception:
            pass
        time.sleep(3)
    sys.exit("app not ready")


def main():
    wait_ready()
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1.5)
        pg.goto(BASE)
        pg.wait_for_selector("#status[data-s='ready']", timeout=60000)
        # 1. invalid input recovery
        pg.click("#run")
        pg.wait_for_timeout(400)
        pg.screenshot(path=OUT / "01_validation_errors.png")
        # 2. unseen case input
        pg.fill("#note", UNSEEN_NOTE)
        pg.fill("#patient-label", "Unseen case A (67M)")
        pg.fill("#paste", UNSEEN_CRITERIA)
        pg.click("#parse-btn")
        pg.fill("#trial-title", "Hypothetical T2DM add-on therapy trial")
        pg.click("#preview-toggle")
        pg.wait_for_timeout(800)
        pg.screenshot(path=OUT / "02_input.png", full_page=True)
        # 3. run + results
        pg.click("#run")
        pg.wait_for_selector(".outcome", timeout=600000)
        pg.wait_for_timeout(500)
        pg.locator("#results").scroll_into_view_if_needed()
        pg.add_style_tag(content=".topbar{position:static!important}")
        pg.wait_for_selector("#status[data-s='ready']", timeout=60000)
        cards = pg.locator(".crit-card")
        if cards.count():
            cards.nth(1).hover()
        pg.wait_for_timeout(300)
        pg.locator("#results").screenshot(path=OUT / "03_results.png")
        # 4. printable report
        href = pg.locator("a:has-text('Open printable report')").get_attribute("href")
        rp = b.new_page(viewport={"width": 900, "height": 1200}, device_scale_factor=1.5)
        rp.goto(BASE + href)
        rp.screenshot(path=OUT / "04_report.png")
        # 5. trial fetch failure is friendly
        pg.click("button[data-tab='nct']")
        pg.fill("#nct", "NCT12")
        pg.click("#nct-btn")
        pg.wait_for_timeout(300)
        pg.locator("section[aria-labelledby='h-crit']").screenshot(path=OUT / "05_nct_validation.png")
        # 6. inference lab
        lab = b.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1.5)
        lab.goto(BASE + "/lab")
        lab.wait_for_selector("#run-dec:not([disabled])", timeout=120000)
        lab.click("#run-dec")
        lab.wait_for_selector("#dec-out .kpis", timeout=600000)
        lab.click("#run-att")
        lab.wait_for_selector("#att-out table", timeout=300000)
        lab.add_style_tag(content=".topbar{position:static!important}")
        lab.wait_for_timeout(500)
        lab.screenshot(path=OUT / "06_lab.png", full_page=True,
                       clip={"x": 0, "y": 0, "width": 1440, "height": 1180})
        b.close()
    from PIL import Image
    for name, h in (("02_input", 1164), ("03_results", 1157), ("04_report", 990)):  # tight crops used in the report
        im = Image.open(OUT / f"{name}.png")
        im.crop((0, 0, im.width, min(h, im.height))).save(OUT / f"{name}_top.png")
    print("screenshots ->", OUT)


if __name__ == "__main__":
    main()
