from ..handles import add_node
from ..workflow_builder import SAMPLERS, SCHEDULERS


class CloudKSamplerGraph:
    """Graph-style cloud KSampler. Drop-in shape match for the local KSampler:
    takes MODEL/CONDITIONING(+/-)/LATENT handles, returns a LATENT handle.
    Does NOT submit anything — only the terminal CloudVAEDecode submits."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":         ("CLOUD_MODEL",),
                "positive":      ("CLOUD_CONDITIONING",),
                "negative":      ("CLOUD_CONDITIONING",),
                "latent_image":  ("CLOUD_LATENT",),
                "seed":          ("INT",   {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps":         ("INT",   {"default": 20, "min": 1, "max": 200}),
                "cfg":           ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name":  (SAMPLERS,),
                "scheduler":     (SCHEDULERS,),
                "denoise":       ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("CLOUD_LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "run"
    CATEGORY = "cloud"

    def run(self, model, positive, negative, latent_image,
            seed, steps, cfg, sampler_name, scheduler, denoise):
        (handle,) = add_node(
            [model, positive, negative, latent_image],
            "KSampler",
            {
                "seed": seed, "steps": steps, "cfg": cfg,
                "sampler_name": sampler_name, "scheduler": scheduler, "denoise": denoise,
                "model":        model.ref,
                "positive":     positive.ref,
                "negative":     negative.ref,
                "latent_image": latent_image.ref,
            },
        )
        return (handle,)
