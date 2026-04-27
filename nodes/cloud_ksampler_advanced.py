from ..handles import add_node
from ..workflow_builder import SAMPLERS, SCHEDULERS

ENABLE_DISABLE = ["enable", "disable"]


class CloudKSamplerAdvanced:
    """Graph-style cloud KSamplerAdvanced. Adds add_noise / start_at_step / end_at_step /
    return_with_leftover_noise — required for chained two-stage sampling (e.g. Wan 2.2 high→low noise)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":         ("CLOUD_MODEL",),
                "positive":      ("CLOUD_CONDITIONING",),
                "negative":      ("CLOUD_CONDITIONING",),
                "latent_image":  ("CLOUD_LATENT",),
                "add_noise":     (ENABLE_DISABLE,),
                "noise_seed":    ("INT",   {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps":         ("INT",   {"default": 4, "min": 1, "max": 200}),
                "cfg":           ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name":  (SAMPLERS,),
                "scheduler":     (SCHEDULERS,),
                "start_at_step": ("INT",   {"default": 0, "min": 0, "max": 10000}),
                "end_at_step":   ("INT",   {"default": 10000, "min": 0, "max": 10000}),
                "return_with_leftover_noise": (ENABLE_DISABLE,),
            }
        }

    RETURN_TYPES = ("CLOUD_LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "run"
    CATEGORY = "cloud"

    def run(self, model, positive, negative, latent_image,
            add_noise, noise_seed, steps, cfg, sampler_name, scheduler,
            start_at_step, end_at_step, return_with_leftover_noise):
        (handle,) = add_node(
            [model, positive, negative, latent_image],
            "KSamplerAdvanced",
            {
                "add_noise":     add_noise,
                "noise_seed":    noise_seed,
                "steps":         steps,
                "cfg":           cfg,
                "sampler_name":  sampler_name,
                "scheduler":     scheduler,
                "start_at_step": start_at_step,
                "end_at_step":   end_at_step,
                "return_with_leftover_noise": return_with_leftover_noise,
                "model":         model.ref,
                "positive":      positive.ref,
                "negative":      negative.ref,
                "latent_image":  latent_image.ref,
            },
        )
        return (handle,)
