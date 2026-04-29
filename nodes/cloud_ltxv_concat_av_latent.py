from ..handles import add_node


class CloudLTXVConcatAVLatent:
    """Joint AV latent: concatenates a video latent and an audio latent into one."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_latent": ("CLOUD_LATENT",),
                "audio_latent": ("CLOUD_LATENT",),
            }
        }

    RETURN_TYPES = ("CLOUD_LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "run"
    CATEGORY = "cloud"

    def run(self, video_latent, audio_latent):
        (handle,) = add_node(
            [video_latent, audio_latent],
            "LTXVConcatAVLatent",
            {"video_latent": video_latent.ref, "audio_latent": audio_latent.ref},
        )
        return (handle,)
