from ..handles import add_node


class CloudLTXVPreprocess:
    """Compresses an image (cloud-side) before feeding into LTX I2V."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":           ("CLOUD_IMAGE",),
                "img_compression": ("INT", {"default": 35, "min": 0, "max": 100}),
            }
        }

    RETURN_TYPES = ("CLOUD_IMAGE",)
    RETURN_NAMES = ("output_image",)
    FUNCTION = "run"
    CATEGORY = "cloud"

    def run(self, image, img_compression):
        (handle,) = add_node(
            [image], "LTXVPreprocess",
            {"image": image.ref, "img_compression": img_compression},
        )
        return (handle,)
