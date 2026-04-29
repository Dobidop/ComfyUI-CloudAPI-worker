from ..handles import add_node


# Hardcoded from /api/object_info — refresh this list as new upscalers are added.
LATENT_UPSCALERS = [
    "hunyuanvideo15_latent_upsampler_1080p.safetensors",
    "hunyuanvideo15_latent_upsampler_720p.safetensors",
    "ltx-2-spatial-upscaler-x2-1.0.safetensors",
    "ltx-2-temporal-upscaler-x2-1.0.safetensors",
    "ltx-2.3-spatial-upscaler-x2-1.0.safetensors",
    "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
]


class CloudLatentUpscaleModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (LATENT_UPSCALERS,),
            }
        }

    RETURN_TYPES = ("CLOUD_LATENT_UPSCALE_MODEL",)
    RETURN_NAMES = ("LATENT_UPSCALE_MODEL",)
    FUNCTION = "run"
    CATEGORY = "cloud"

    def run(self, model_name):
        (handle,) = add_node([], "LatentUpscaleModelLoader", {"model_name": model_name})
        return (handle,)
