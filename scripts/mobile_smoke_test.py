from __future__ import annotations

import json
from pathlib import Path

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "qa_artifacts"
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET_URL = "http://localhost:8501"
SCREENSHOT_PATH = OUTPUT_DIR / "mobile-home.png"
REPORT_PATH = OUTPUT_DIR / "mobile-smoke.json"


def main() -> None:
    with sync_playwright() as playwright:
        device = playwright.devices["iPhone 13"]
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(**device)
        page = context.new_page()

        page.goto(TARGET_URL, wait_until="networkidle", timeout=60000)

        checks = {
            "hero_heading": "Read your digital rhythm, then simulate a better one.",
            "submit_button": "Analyze My Lifestyle",
            "why_heading": "Why This Is Happening",
            "scenario_heading": "Scenario Lab",
            "next_steps_heading": "What Should You Do Next?",
        }

        results: dict[str, object] = {
            "url": TARGET_URL,
            "device": "iPhone 13",
            "viewport": device["viewport"],
            "checks": {},
        }

        for key, text in checks.items():
            try:
                page.get_by_text(text, exact=False).first.wait_for(timeout=20000)
                results["checks"][key] = True
            except PlaywrightTimeoutError:
                results["checks"][key] = False

        page.screenshot(path=str(SCREENSHOT_PATH), full_page=True)
        results["title"] = page.title()
        results["screenshot"] = str(SCREENSHOT_PATH)
        REPORT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

        browser.close()

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
