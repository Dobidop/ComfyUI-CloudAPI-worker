"""Non-terminal VAE Decode. Returns a CLOUD_IMAGE handle (a ref into the assembled
workflow); subsequent cloud nodes consume it without forcing a submit/download.

Use a Cloud Fetch Images terminal afterwards to materialize the CLOUD_IMAGE into
a local IMAGE tensor."""
from ..handles import add_node


class CloudVAEDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("CLOUD_LATENT",),
                "vae":     ("CLOUD_VAE",),
            }
        }

    RETURN_TYPES = ("CLOUD_IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    CATEGORY = "cloud"

    def run(self, samples, vae):
        (handle,) = add_node(
            [samples, vae],
            "VAEDecode",
            {"samples": samples.ref, "vae": vae.ref},
        )
        return (handle,)
