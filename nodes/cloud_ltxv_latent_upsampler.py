from ..handles import add_node


class CloudLTXVLatentUpsampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples":       ("CLOUD_LATENT",),
                "upscale_model": ("CLOUD_LATENT_UPSCALE_MODEL",),
                "vae":           ("CLOUD_VAE",),
            }
        }

    RETURN_TYPES = ("CLOUD_LATENT",)
    RETURN_NAMES = ("LATENT",)
    FUNCTION = "run"
    CATEGORY = "cloud"

    def run(self, samples, upscale_model, vae):
        (handle,) = add_node(
            [samples, upscale_model, vae],
            "LTXVLatentUpsampler",
            {"samples": samples.ref, "upscale_model": upscale_model.ref, "vae": vae.ref},
        )
        return (handle,)
