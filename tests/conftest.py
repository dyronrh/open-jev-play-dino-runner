import os

import pytest


def _chromium_available() -> bool:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return False
    try:
        with sync_playwright() as p:
            kwargs = {"executable_path": os.environ["CHROMIUM_PATH"]} if os.environ.get("CHROMIUM_PATH") else {}
            p.chromium.launch(**kwargs).close()
        return True
    except Exception:
        return False


CHROMIUM = _chromium_available()
needs_chromium = pytest.mark.skipif(not CHROMIUM, reason="needs Playwright Chromium (playwright install chromium, or CHROMIUM_PATH)")
