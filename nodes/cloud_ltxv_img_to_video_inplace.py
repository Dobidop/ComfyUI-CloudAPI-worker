from ..handles import add_node


class CloudLTXVImgToVideoInplace:
    """LTX I2V in-place conditioning — bakes a start image into a latent."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae":      ("CLOUD_VAE",),
                "image":    ("CLOUD_IMAGE",),
                "latent":   ("CLOUD_LATENT",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
                "bypass":   ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("CLOUD_LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "run"
    CATEGORY = "cloud"

    def run(self, vae, image, latent, strength, bypass):
        (handle,) = add_node(
            [vae, image, latent], "LTXVImgToVideoInplace",
            {"vae": vae.ref, "image": image.ref, "latent": latent.ref,
             "strength": strength, "bypass": bypass},
        )
        return (handle,)
