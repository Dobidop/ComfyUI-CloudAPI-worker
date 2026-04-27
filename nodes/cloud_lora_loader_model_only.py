from ..config import get_cached_models
from ..handles import add_node


def _lora_choices():
    cached = get_cached_models("loras")
    return cached if cached else ["<run Cloud List Models (loras) to populate>"]


class CloudLoraLoaderModelOnly:
    """Model-only LoRA — applies a LoRA to MODEL only, no CLIP path. Used in many video workflows."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":          ("CLOUD_MODEL",),
                "lora_name":      (_lora_choices(),),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("CLOUD_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "run"
    CATEGORY = "cloud"

    def run(self, model, lora_name, strength_model):
        if lora_name.startswith("<"):
            raise RuntimeError("No LoRA selected. Run Cloud List Models (folder=loras) and restart ComfyUI.")
        (handle,) = add_node(
            [model],
            "LoraLoaderModelOnly",
            {"model": model.ref, "lora_name": lora_name, "strength_model": strength_model},
        )
        return (handle,)
