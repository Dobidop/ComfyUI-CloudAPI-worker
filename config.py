import json
import os
import shutil
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(_HERE, "config.json")
EXAMPLE_PATH = os.path.join(_HERE, "config.json.example")

CACHE_TTL_SECONDS = 24 * 60 * 60


def _ensure_config_exists():
    if not os.path.exists(CONFIG_PATH):
        if os.path.exists(EXAMPLE_PATH):
            shutil.copy(EXAMPLE_PATH, CONFIG_PATH)
            print(f"[CloudAPI] Created config.json from example. Edit {CONFIG_PATH} and add your API key.")
        else:
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump({"api_key": "", "base_url": "https://cloud.comfy.org",
                           "cached_checkpoints": [], "cached_checkpoints_fetched_at": None}, f, indent=2)


def get_config():
    _ensure_config_exists()
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(cfg):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def get_api_key():
    cfg = get_config()
    key = cfg.get("api_key", "").strip()
    if not key or key.startswith("PASTE_"):
        raise RuntimeError(
            f"Cloud API key not set. Edit {CONFIG_PATH} and paste a key from "
            "https://platform.comfy.org/profile/api-keys"
        )
    return key


def get_base_url():
    return get_config().get("base_url", "https://cloud.comfy.org").rstrip("/")


def get_cached_models(folder):
    """Return cached model name list for the given folder. Empty if never fetched.
    Reads from cached_models[folder]; falls back to legacy cached_checkpoints for 'checkpoints'."""
    cfg = get_config()
    cache = cfg.get("cached_models") or {}
    if folder in cache:
        return list(cache[folder])
    if folder == "checkpoints":
        return list(cfg.get("cached_checkpoints") or [])
    return []


def update_cached_models(folder, names):
    cfg = get_config()
    cache = cfg.get("cached_models") or {}
    cache[folder] = list(names)
    cfg["cached_models"] = cache
    fetched = cfg.get("cached_models_fetched_at") or {}
    fetched[folder] = int(time.time())
    cfg["cached_models_fetched_at"] = fetched
    if folder == "checkpoints":
        cfg["cached_checkpoints"] = list(names)
        cfg["cached_checkpoints_fetched_at"] = int(time.time())
    save_config(cfg)


def get_cached_checkpoints():
    return get_cached_models("checkpoints")


def update_cached_checkpoints(names):
    update_cached_models("checkpoints", names)
