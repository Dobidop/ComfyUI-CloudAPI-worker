from ..handles import add_node


class CloudLTXVSeparateAVLatent:
    """Splits a joint AV latent back into separate video and audio latents."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "av_latent": ("CLOUD_LATENT",),
            }
        }

    RETURN_TYPES = ("CLOUD_LATENT", "CLOUD_LATENT")
    RETURN_NAMES = ("video_latent", "audio_latent")
    FUNCTION = "run"
    CATEGORY = "cloud"

    def run(self, av_latent):
        v, a = add_node(
            [av_latent], "LTXVSeparateAVLatent",
            {"av_latent": av_latent.ref},
            num_outputs=2,
        )
        return (v, a)
