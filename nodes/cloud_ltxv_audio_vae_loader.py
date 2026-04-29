from ..handles import add_node


class CloudLTXVAudioVAELoader:
    """LTX 2.0 audio VAE loader — outputs CLOUD_AUDIO_VAE. Treated as a
    separate handle type so it can't be wired into nodes expecting a regular
    image VAE. Filename is a STRING because the cloud's options list is huge."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": ("STRING", {"default": "stable-audio-open-1.0.safetensors"}),
            }
        }

    RETURN_TYPES = ("CLOUD_AUDIO_VAE",)
    RETURN_NAMES = ("Audio VAE",)
    FUNCTION = "run"
    CATEGORY = "cloud"

    def run(self, ckpt_name):
        (handle,) = add_node([], "LTXVAudioVAELoader", {"ckpt_name": ckpt_name})
        return (handle,)
