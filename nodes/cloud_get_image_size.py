from ..handles import add_node


class CloudGetImageSize:
    """Returns (width, height, batch_size) of a CLOUD_IMAGE — INTs flow as plain
    Python ints out of the cloud and back in. Used heavily inside LTX templates
    to drive resize / aspect calculations."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("CLOUD_IMAGE",),
            }
        }

    RETURN_TYPES = ("CLOUD_INT", "CLOUD_INT", "CLOUD_INT")
    RETURN_NAMES = ("width", "height", "batch_size")
    FUNCTION = "run"
    CATEGORY = "cloud"

    def run(self, image):
        w, h, b = add_node(
            [image], "GetImageSize",
            {"image": image.ref},
            num_outputs=3,
        )
        return (w, h, b)
