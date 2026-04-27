from ..config import get_cached_models
from ..handles import add_node


def _lora_choices():
    cached = get_cached_models("loras")
    return cached if cached else ["<run Cloud List Models (loras) to populate>"]


class CloudLoraLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":           ("CLOUD_MODEL",),
                "clip":            ("CLOUD_CLIP",),
                "lora_name":       (_lora_choices(),),
                "strength_model":  ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "strength_clip":   ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("CLOUD_MODEL", "CLOUD_CLIP")
    RETURN_NAMES = ("model", "clip")
    FUNCTION = "run"
    CATEGORY = "cloud"

    def run(self, model, clip, lora_name, strength_model, strength_clip):
        if lora_name.startswith("<"):
            raise RuntimeError("No LoRA selected. Run Cloud List Models (folder=loras) and restart ComfyUI.")
        m_handle, c_handle = add_node(
            [model, clip],
            "LoraLoader",
            {
                "model": model.ref,
                "clip":  clip.ref,
                "lora_name":      lora_name,
                "strength_model": strength_model,
                "strength_clip":  strength_clip,
            },
            num_outputs=2,
        )
        return (m_handle, c_handle)
