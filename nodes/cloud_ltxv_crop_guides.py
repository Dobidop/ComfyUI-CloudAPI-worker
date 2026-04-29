from ..handles import add_node


class CloudLTXVCropGuides:
    """Crops conditioning guide frames out of an LTX latent. Returns updated
    pos/neg conditioning plus the cropped latent."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CLOUD_CONDITIONING",),
                "negative": ("CLOUD_CONDITIONING",),
                "latent":   ("CLOUD_LATENT",),
            }
        }

    RETURN_TYPES = ("CLOUD_CONDITIONING", "CLOUD_CONDITIONING", "CLOUD_LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "run"
    CATEGORY = "cloud"

    def run(self, positive, negative, latent):
        pos, neg, lat = add_node(
            [positive, negative, latent],
            "LTXVCropGuides",
            {"positive": positive.ref, "negative": negative.ref, "latent": latent.ref},
            num_outputs=3,
        )
        return (pos, neg, lat)
