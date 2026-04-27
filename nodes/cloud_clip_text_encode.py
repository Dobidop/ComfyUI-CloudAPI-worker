from ..handles import add_node


class CloudCLIPTextEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLOUD_CLIP",),
                "text": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("CLOUD_CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "run"
    CATEGORY = "cloud"

    def run(self, clip, text):
        (handle,) = add_node(
            [clip],
            "CLIPTextEncode",
            {"clip": clip.ref, "text": text},
        )
        return (handle,)
