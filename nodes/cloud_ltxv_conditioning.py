from ..handles import add_node


class CloudLTXVConditioning:
    """Sets the frame_rate field on LTX conditioning."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive":   ("CLOUD_CONDITIONING",),
                "negative":   ("CLOUD_CONDITIONING",),
                "frame_rate": ("FLOAT", {"default": 25.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("CLOUD_CONDITIONING", "CLOUD_CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "run"
    CATEGORY = "cloud"

    def run(self, positive, negative, frame_rate):
        pos, neg = add_node(
            [positive, negative],
            "LTXVConditioning",
            {"positive": positive.ref, "negative": negative.ref, "frame_rate": frame_rate},
            num_outputs=2,
        )
        return (pos, neg)
