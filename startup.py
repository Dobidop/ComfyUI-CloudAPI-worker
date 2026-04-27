"""Background prefetch of model lists at ComfyUI startup.

Runs in a daemon thread so it never blocks startup or crashes ComfyUI if the API
is unreachable / key missing. Updates the on-disk cache that drives node dropdowns."""
import threading
import time

from .api_client import ComfyCloudClient, CloudAPIError
from .config import (
    CACHE_TTL_SECONDS,
    get_api_key,
    get_base_url,
    get_config,
    update_cached_models,
)

# Folders prefetched on startup. Add more here if you want extra dropdowns
# populated automatically (vae, controlnet, etc.).
PREFETCH_FOLDERS = [
    "checkpoints",
    "loras",
    "vae",
    "diffusion_models",
    "text_encoders",
    "clip_vision",
]


def _folder_cache_age(cfg, folder):
    fetched = (cfg.get("cached_models_fetched_at") or {}).get(folder)
    if not fetched:
        return None
    return time.time() - fetched


def _refresh_one(client, folder):
    try:
        models = client.list_models(folder)
        names = [m.get("name") for m in (models or []) if isinstance(m, dict) and m.get("name")]
        update_cached_models(folder, names)
        print(f"[CloudAPI] Prefetched {len(names)} {folder}.")
    except CloudAPIError as e:
        print(f"[CloudAPI] Prefetch of '{folder}' failed: {e}")
    except Exception as e:
        print(f"[CloudAPI] Unexpected error prefetching '{folder}': {e}")


def _prefetch_worker(force=False):
    try:
        api_key = get_api_key()
    except Exception as e:
        print(f"[CloudAPI] Skipping startup model prefetch: {e}")
        return

    cfg = get_config()
    client = ComfyCloudClient(api_key, get_base_url())

    for folder in PREFETCH_FOLDERS:
        age = _folder_cache_age(cfg, folder)
        if not force and age is not None and age < CACHE_TTL_SECONDS:
            print(f"[CloudAPI] Cache for '{folder}' is fresh ({int(age)}s old); skipping.")
            continue
        _refresh_one(client, folder)


def kick_off_prefetch(force=False):
    t = threading.Thread(target=_prefetch_worker, kwargs={"force": force},
                         name="CloudAPI-Prefetch", daemon=True)
    t.start()
    return t
