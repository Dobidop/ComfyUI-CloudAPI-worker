"""Non-terminal VAE Decode. Returns a CLOUD_IMAGE handle (a ref into the assembled
workflow), letting subsequent cloud nodes consume it without forcing a submit/download.

Use this when the decoded image will be consumed by another cloud node
(CloudCreateVideo, future cloud upscalers, etc.). Use the existing terminal
CloudVAEDecode when you want the image batch back as a real local IMAGE tensor."""
from ..handles import add_node


class CloudVAEDecodeRef:
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
