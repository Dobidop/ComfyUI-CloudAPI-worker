from ..config import get_cached_models
from ..handles import add_node
from ..workflow_builder import CLIP_TYPES


def _clip_choices():
    # ComfyUI uses "text_encoders" in newer builds, "clip" in older ones.
    cached = get_cached_models("text_encoders") or get_cached_models("clip")
    return cached if cached else ["<run Cloud List Models (text_encoders) to populate>"]


class CloudCLIPLoader:
    """Single text-encoder loader. Use CloudDualCLIPLoader for SDXL/SD3/Flux/HunyuanVideo."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_name": (_clip_choices(),),
                "type":      (CLIP_TYPES,),
            },
            "optional": {
                "device": (["default", "cpu"],),
            },
        }

    RETURN_TYPES = ("CLOUD_CLIP",)
    RETURN_NAMES = ("clip",)
    FUNCTION = "run"
    CATEGORY = "cloud"

    def run(self, clip_name, type, device="default"):
        if clip_name.startswith("<"):
            raise RuntimeError(
                "No text encoder selected. Run Cloud List Models (folder=text_encoders) and restart ComfyUI."
            )
        (handle,) = add_node(
            [], "CLIPLoader",
            {"clip_name": clip_name, "type": type, "device": device},
        )
        return (handle,)
