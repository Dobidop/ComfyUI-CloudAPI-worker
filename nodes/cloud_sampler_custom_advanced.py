from ..handles import add_node


class CloudSamplerCustomAdvanced:
    """SamplerCustomAdvanced — flexible sampling with explicit Noise/Guider/Sampler/Sigmas pieces.
    Returns two latents: `output` (the diffused result) and `denoised_output` (the model's
    final denoised prediction). LTX 2.0 uses the first; advanced workflows may use both."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise":        ("CLOUD_NOISE",),
                "guider":       ("CLOUD_GUIDER",),
                "sampler":      ("CLOUD_SAMPLER",),
                "sigmas":       ("CLOUD_SIGMAS",),
                "latent_image": ("CLOUD_LATENT",),
            }
        }

    RETURN_TYPES = ("CLOUD_LATENT", "CLOUD_LATENT")
    RETURN_NAMES = ("output", "denoised_output")
    FUNCTION = "run"
    CATEGORY = "cloud"

    def run(self, noise, guider, sampler, sigmas, latent_image):
        out, denoised = add_node(
            [noise, guider, sampler, sigmas, latent_image],
            "SamplerCustomAdvanced",
            {
                "noise":        noise.ref,
                "guider":       guider.ref,
                "sampler":      sampler.ref,
                "sigmas":       sigmas.ref,
                "latent_image": latent_image.ref,
            },
            num_outputs=2,
        )
        return (out, denoised)
