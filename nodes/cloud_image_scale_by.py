from ..handles import add_node

UPSCALE_METHODS = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]


class CloudImageScaleBy:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":          ("CLOUD_IMAGE",),
                "upscale_method": (UPSCALE_METHODS,),
                "scale_by":       ("FLOAT", {"default": 1.0, "min": 0.01, "max": 8.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("CLOUD_IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "cloud"

    def run(self, image, upscale_method, scale_by):
        (handle,) = add_node(
            [image], "ImageScaleBy",
            {"image": image.ref, "upscale_method": upscale_method, "scale_by": scale_by},
        )
        return (handle,)
