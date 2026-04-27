from ..handles import add_node


class CloudEmptyLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width":      ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "height":     ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 16}),
            }
        }

    RETURN_TYPES = ("CLOUD_LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "run"
    CATEGORY = "cloud"

    def run(self, width, height, batch_size):
        (handle,) = add_node(
            [],
            "EmptyLatentImage",
            {"width": width, "height": height, "batch_size": batch_size},
        )
        return (handle,)
