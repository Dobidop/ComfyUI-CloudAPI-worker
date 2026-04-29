from ..handles import add_node


class CloudManualSigmas:
    """Comma-separated sigma sequence (e.g. "1.0, 0.6, 0.3, 0.1, 0")."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sigmas": ("STRING", {"default": "1, 0.5", "multiline": False}),
            }
        }

    RETURN_TYPES = ("CLOUD_SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    FUNCTION = "run"
    CATEGORY = "cloud"

    def run(self, sigmas):
        (handle,) = add_node([], "ManualSigmas", {"sigmas": sigmas})
        return (handle,)
