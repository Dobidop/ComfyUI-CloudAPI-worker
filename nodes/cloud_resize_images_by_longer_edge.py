from ..handles import add_node


class CloudResizeImagesByLongerEdge:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images":      ("CLOUD_IMAGE",),
                "longer_edge": ("INT", {"default": 1024, "min": 1, "max": 8192}),
            }
        }

    RETURN_TYPES = ("CLOUD_IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    CATEGORY = "cloud"

    def run(self, images, longer_edge):
        (handle,) = add_node(
            [images], "ResizeImagesByLongerEdge",
            {"images": images.ref, "longer_edge": longer_edge},
        )
        return (handle,)
