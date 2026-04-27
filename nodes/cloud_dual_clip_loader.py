from ..config import get_cached_models
from ..handles import add_node
from ..workflow_builder import DUAL_CLIP_TYPES


def _clip_choices():
    cached = get_cached_models("text_encoders") or get_cached_models("clip")
    return cached if cached else ["<run Cloud List Models (text_encoders) to populate>"]


class CloudDualCLIPLoader:
    """Two-text-encoder loader. Used by SDXL, SD3, Flux, HunyuanVideo, HiDream."""

    @classmethod
    def INPUT_TYPES(cls):
        choices = _clip_choices()
        return {
            "required": {
                "clip_name1": (choices,),
                "clip_name2": (choices,),
                "type":       (DUAL_CLIP_TYPES,),
            },
            "optional": {
                "device": (["default", "cpu"],),
            },
        }

    RETURN_TYPES = ("CLOUD_CLIP",)
    RETURN_NAMES = ("clip",)
    FUNCTION = "run"
    CATEGORY = "cloud"

    def run(self, clip_name1, clip_name2, type, device="default"):
        if clip_name1.startswith("<") or clip_name2.startswith("<"):
            raise RuntimeError(
                "Text encoders not selected. Run Cloud List Models (folder=text_encoders) and restart ComfyUI."
            )
        (handle,) = add_node(
            [],
            "DualCLIPLoader",
            {"clip_name1": clip_name1, "clip_name2": clip_name2, "type": type, "device": device},
        )
        return (handle,)
