from ..handles import add_node
from ..workflow_builder import VIDEO_LATENT_FORMATS


_FORMAT_TO_CLASS = {
    "hunyuan_wan": "EmptyHunyuanLatentVideo",
    "ltxv":        "EmptyLTXVLatentVideo",
}


class CloudEmptyLatentVideo:
    """Empty video latent. 'format' picks the right ComfyUI node for the model family."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "format":     (VIDEO_LATENT_FORMATS,),
                "width":      ("INT", {"default": 832,  "min": 64, "max": 8192, "step": 8}),
                "height":     ("INT", {"default": 480,  "min": 64, "max": 8192, "step": 8}),
                "length":     ("INT", {"default": 33,   "min": 1,  "max": 4096}),
                "batch_size": ("INT", {"default": 1,    "min": 1,  "max": 16}),
            }
        }

    RETURN_TYPES = ("CLOUD_LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "run"
    CATEGORY = "cloud"

    def run(self, format, width, height, length, batch_size):
        class_type = _FORMAT_TO_CLASS[format]
        (handle,) = add_node(
            [], class_type,
            {"width": width, "height": height, "length": length, "batch_size": batch_size},
        )
        return (handle,)
