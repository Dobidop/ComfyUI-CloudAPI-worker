from ..handles import add_node


class CloudEmptyImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width":      ("INT", {"default": 512, "min": 1, "max": 16384, "step": 1}),
                "height":     ("INT", {"default": 512, "min": 1, "max": 16384, "step": 1}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "color":      ("INT", {"default": 0, "min": 0, "max": 16777215, "step": 1, "display": "color"}),
            }
        }

    RETURN_TYPES = ("CLOUD_IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "cloud"

    def run(self, width, height, batch_size, color):
        (handle,) = add_node(
            [], "EmptyImage",
            {"width": width, "height": height, "batch_size": batch_size, "color": color},
        )
        return (handle,)
