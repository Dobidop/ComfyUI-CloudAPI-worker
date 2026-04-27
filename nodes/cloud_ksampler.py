"""Quick all-in-one cloud sampler. Builds a complete checkpoint→encode→ksample→decode workflow
internally and submits it. Kept for users who want a single-node txt2img/img2img call;
for fully wired graphs (LoRAs, separate CLIP, etc.) use the graph-style nodes instead."""
import hashlib
import json

from ..api_client import ComfyCloudClient
from ..config import get_api_key, get_base_url, get_cached_checkpoints
from ..image_utils import tensor_to_pil
from ..run_helpers import submit_and_collect_images
from ..workflow_builder import SAMPLERS, SCHEDULERS, build_img2img, build_text2img


def _checkpoint_choices():
    cached = get_cached_checkpoints()
    return cached if cached else ["<run Cloud List Models to populate>"]


class CloudKSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name":   (_checkpoint_choices(),),
                "positive":    ("STRING", {"multiline": True, "default": ""}),
                "negative":    ("STRING", {"multiline": True, "default": ""}),
                "width":       ("INT",   {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "height":      ("INT",   {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "batch_size":  ("INT",   {"default": 1, "min": 1, "max": 16}),
                "seed":        ("INT",   {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps":       ("INT",   {"default": 20, "min": 1, "max": 200}),
                "cfg":         ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (SAMPLERS,),
                "scheduler":    (SCHEDULERS,),
                "denoise":     ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "poll_interval": ("FLOAT", {"default": 3.0, "min": 0.5, "max": 60.0, "step": 0.5}),
                "timeout":     ("INT",   {"default": 600, "min": 10, "max": 3600}),
            },
            "optional": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "prompt_id")
    FUNCTION = "run"
    CATEGORY = "cloud"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        payload = {k: v for k, v in kwargs.items() if k != "image"}
        if "image" in kwargs and kwargs["image"] is not None:
            payload["_has_image"] = True
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()

    def run(self, ckpt_name, positive, negative, width, height, batch_size,
            seed, steps, cfg, sampler_name, scheduler, denoise,
            poll_interval, timeout, image=None):

        if ckpt_name.startswith("<"):
            raise RuntimeError(
                "No checkpoint selected. Run a 'Cloud List Models' node with folder=checkpoints, "
                "then restart ComfyUI to populate the dropdown."
            )

        if image is not None:
            client = ComfyCloudClient(get_api_key(), get_base_url())
            cloud_filename = client.upload_image(tensor_to_pil(image), filename="cloud_ksampler_input.png")
            print(f"[CloudAPI] Uploaded input image as '{cloud_filename}'")
            workflow = build_img2img(
                ckpt=ckpt_name, positive=positive, negative=negative,
                cloud_image_filename=cloud_filename,
                seed=seed, steps=steps, cfg=cfg,
                sampler_name=sampler_name, scheduler=scheduler, denoise=denoise,
            )
        else:
            workflow = build_text2img(
                ckpt=ckpt_name, positive=positive, negative=negative,
                width=width, height=height, batch_size=batch_size,
                seed=seed, steps=steps, cfg=cfg,
                sampler_name=sampler_name, scheduler=scheduler, denoise=denoise,
            )

        images, prompt_id = submit_and_collect_images(workflow, poll_interval=poll_interval, timeout=timeout)
        return (images, prompt_id)
