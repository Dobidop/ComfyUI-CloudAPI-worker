from ..handles import add_node


class CloudCFGGuider:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":    ("CLOUD_MODEL",),
                "positive": ("CLOUD_CONDITIONING",),
                "negative": ("CLOUD_CONDITIONING",),
                "cfg":      ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
            }
        }

    RETURN_TYPES = ("CLOUD_GUIDER",)
    RETURN_NAMES = ("guider",)
    FUNCTION = "run"
    CATEGORY = "cloud"

    def run(self, model, positive, negative, cfg):
        (handle,) = add_node(
            [model, positive, negative],
            "CFGGuider",
            {"model": model.ref, "positive": positive.ref, "negative": negative.ref, "cfg": cfg},
        )
        return (handle,)
