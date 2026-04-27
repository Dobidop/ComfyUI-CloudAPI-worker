from ..handles import add_node


class CloudModelSamplingSD3:
    """Sets the flow-matching 'shift' parameter on a model. Used by SD3, Flux, Wan, etc."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("CLOUD_MODEL",),
                "shift": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("CLOUD_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "run"
    CATEGORY = "cloud"

    def run(self, model, shift):
        (handle,) = add_node(
            [model], "ModelSamplingSD3",
            {"model": model.ref, "shift": shift},
        )
        return (handle,)
