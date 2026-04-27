from ..api_client import ComfyCloudClient
from ..config import get_api_key, get_base_url, update_cached_models
from ..workflow_builder import MODEL_FOLDERS


class CloudListModels:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder": (MODEL_FOLDERS, {"default": "checkpoints"}),
            },
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("model_list", "count")
    FUNCTION = "run"
    CATEGORY = "cloud"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, folder):
        # Always re-run so users can refresh the cache by re-queueing.
        import time
        return f"{folder}:{time.time()}"

    def run(self, folder):
        client = ComfyCloudClient(get_api_key(), get_base_url())
        models = client.list_models(folder)

        names = []
        for m in models or []:
            if isinstance(m, dict):
                n = m.get("name")
                if n:
                    names.append(n)
            elif isinstance(m, str):
                names.append(m)

        update_cached_models(folder, names)
        print(f"[CloudAPI] Cached {len(names)} {folder}. Restart ComfyUI to refresh dropdowns.")

        text = "\n".join(names) if names else "(no models found)"
        print(f"[CloudAPI] {folder}: {len(names)} model(s)\n{text}")
        return (text, len(names))
