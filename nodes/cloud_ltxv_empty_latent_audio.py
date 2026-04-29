from ..handles import add_node


class CloudLTXVEmptyLatentAudio:
    """Empty audio latent sized for the given frame count + frame rate."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames_number": ("INT", {"default": 97, "min": 1, "max": 1000, "step": 1}),
                "frame_rate":    ("INT", {"default": 25, "min": 1, "max": 1000, "step": 1}),
                "batch_size":    ("INT", {"default": 1, "min": 1, "max": 4096}),
                "audio_vae":     ("CLOUD_AUDIO_VAE",),
            }
        }

    RETURN_TYPES = ("CLOUD_LATENT",)
    RETURN_NAMES = ("Latent",)
    FUNCTION = "run"
    CATEGORY = "cloud"

    def run(self, frames_number, frame_rate, batch_size, audio_vae):
        (handle,) = add_node(
            [audio_vae], "LTXVEmptyLatentAudio",
            {"frames_number": frames_number, "frame_rate": frame_rate,
             "batch_size": batch_size, "audio_vae": audio_vae.ref},
        )
        return (handle,)
