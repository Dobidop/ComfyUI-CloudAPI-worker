from ..config import get_cached_models
from ..handles import add_node


def _vae_choices():
    cached = get_cached_models("vae")
    return cached if cached else ["<run Cloud List Models (vae) to populate>"]


class CloudVAELoader:
    """Loads a standalone VAE file (separate from a checkpoint bundle)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae_name": (_vae_choices(),),
            }
        }

    RETURN_TYPES = ("CLOUD_VAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "run"
    CATEGORY = "cloud"

    def run(self, vae_name):
        if vae_name.startswith("<"):
            raise RuntimeError(
                "No VAE selected. Run Cloud List Models (folder=vae) and restart ComfyUI."
            )
        (handle,) = add_node([], "VAELoader", {"vae_name": vae_name})
        return (handle,)
