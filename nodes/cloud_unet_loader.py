from ..config import get_cached_models
from ..handles import add_node
from ..workflow_builder import UNET_WEIGHT_DTYPES


def _unet_choices():
    cached = get_cached_models("diffusion_models")
    return cached if cached else ["<run Cloud List Models (diffusion_models) to populate>"]


class CloudUNETLoader:
    """Loads a bare diffusion model file (used by Wan, Hunyuan, LTXV, Flux, etc.)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unet_name":    (_unet_choices(),),
                "weight_dtype": (UNET_WEIGHT_DTYPES,),
            }
        }

    RETURN_TYPES = ("CLOUD_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "run"
    CATEGORY = "cloud"

    def run(self, unet_name, weight_dtype):
        if unet_name.startswith("<"):
            raise RuntimeError(
                "No diffusion model selected. Run Cloud List Models (folder=diffusion_models) "
                "and restart ComfyUI."
            )
        (handle,) = add_node(
            [],
            "UNETLoader",
            {"unet_name": unet_name, "weight_dtype": weight_dtype},
        )
        return (handle,)
