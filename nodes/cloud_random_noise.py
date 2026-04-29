from ..handles import add_node


class CloudRandomNoise:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("CLOUD_NOISE",)
    RETURN_NAMES = ("noise",)
    FUNCTION = "run"
    CATEGORY = "cloud"

    def run(self, noise_seed):
        (handle,) = add_node([], "RandomNoise", {"noise_seed": noise_seed})
        return (handle,)
