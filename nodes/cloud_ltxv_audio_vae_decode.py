from ..handles import add_node


class CloudLTXVAudioVAEDecode:
    """Decodes an audio latent into a CLOUD_AUDIO ref."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples":   ("CLOUD_LATENT",),
                "audio_vae": ("CLOUD_AUDIO_VAE",),
            }
        }

    RETURN_TYPES = ("CLOUD_AUDIO",)
    RETURN_NAMES = ("Audio",)
    FUNCTION = "run"
    CATEGORY = "cloud"

    def run(self, samples, audio_vae):
        (handle,) = add_node(
            [samples, audio_vae], "LTXVAudioVAEDecode",
            {"samples": samples.ref, "audio_vae": audio_vae.ref},
        )
        return (handle,)
