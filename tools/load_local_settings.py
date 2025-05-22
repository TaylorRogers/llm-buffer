# tools/load_local_settings.py
import os
import json
from pathlib import Path

def load_local_settings():
    settings_path = Path(__file__).resolve().parent.parent / "local.settings.json"
    if settings_path.exists():
        with open(settings_path) as f:
            local_settings = json.load(f)
            for key, value in local_settings.get("Values", {}).items():
                os.environ.setdefault(key, value)
