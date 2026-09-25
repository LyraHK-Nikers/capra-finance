"""Open the deployed Streamlit app in a headless browser so it doesn't fall asleep.

Streamlit Community Cloud puts apps to sleep after ~12 hours without visitors. A plain
HTTP request doesn't count as a visit (the app only counts real browser sessions), so
this loads the page like a person would, clicks "Yes, get this app back up!" if the app
has already gone to sleep, waits for the app itself to render, then stays connected for
a bit so the visit registers.

Exits non-zero if the app never renders, so a failed GitHub Actions run means the app
is actually down (not just asleep).
"""
import os
import re
import sys
import time

from playwright.sync_api import sync_playwright

URL = os.environ.get("APP_URL", "https://capra-finance-ftwkn5swmhbya78nolgcja.streamlit.app/")
MARKER = "CAPRA Finance"        # text the app renders (header / password screen)
BOOT_TIMEOUT = 240              # seconds to wait for a cold start
STAY_SECONDS = 30               # stay connected after it renders


def app_rendered(page) -> bool:
    for frame in page.frames:
        try:
            if MARKER in frame.locator("body").inner_text(timeout=2_000):
                return True
        except Exception:
            pass
    return False


with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto(URL, wait_until="domcontentloaded", timeout=120_000)

    wake = page.get_by_role("button", name=re.compile("get this app back up", re.I))
    try:
        wake.wait_for(state="visible", timeout=15_000)
        print("App was asleep - clicking the wake button")
        wake.click()
    except Exception:
        print("No wake button - app was awake or is already waking")

    deadline = time.time() + BOOT_TIMEOUT
    rendered = False
    while time.time() < deadline:
        if app_rendered(page):
            rendered = True
            break
        page.wait_for_timeout(5_000)

    if rendered:
        print(f"App rendered - staying connected {STAY_SECONDS}s")
        page.wait_for_timeout(STAY_SECONDS * 1_000)
    else:
        print(f"App did not render within {BOOT_TIMEOUT}s")

    browser.close()
    sys.exit(0 if rendered else 1)
