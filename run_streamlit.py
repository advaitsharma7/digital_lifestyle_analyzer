from __future__ import annotations

import os
import sys
from pathlib import Path

from src import bootstrap  # noqa: F401

from streamlit import config as _config
from streamlit.runtime.credentials import check_credentials
from streamlit.web import bootstrap as stbootstrap


ROOT_DIR = Path(__file__).resolve().parent


if __name__ == "__main__":
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
    os.environ.setdefault("STREAMLIT_SERVER_SHOW_EMAIL_PROMPT", "false")
    port = int(os.environ.get("PORT", os.environ.get("STREAMLIT_SERVER_PORT", "8501")))
    address = os.environ.get("STREAMLIT_SERVER_ADDRESS", "0.0.0.0")
    main_script_path = str((ROOT_DIR / "app.py").resolve())
    flag_options = {
        "global.developmentMode": False,
        "server.address": address,
        "server.port": port,
        "server.headless": True,
        "server.showEmailPrompt": False,
        "browser.gatherUsageStats": False,
    }
    _config._main_script_path = main_script_path
    stbootstrap.load_config_options(flag_options=flag_options)
    check_credentials()
    stbootstrap.run(main_script_path, False, [], flag_options)
