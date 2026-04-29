import hashlib
import json

from ..handles import fresh_id, merge_node_dicts
from ..run_helpers import submit_and_collect_images


class CloudVAEDecode:
    """Terminal node. Appends VAEDecode + SaveImage to the assembled workflow,
    submits to /api/prompt, polls, and downloads the resulting images."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples":       ("CLOUD_LATENT",),
                "vae":           ("CLOUD_VAE",),
                "poll_interval": ("FLOAT", {"default": 3.0, "min": 0.5, "max": 60.0, "step": 0.5}),
                "timeout":       ("INT",   {"default": 600, "min": 10, "max": 3600}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "prompt_id")
    FUNCTION = "run"
    CATEGORY = "cloud"

    @classmethod
    def IS_CHANGED(cls, samples=None, vae=None, poll_interval=3.0, timeout=600):
        # Hash the assembled workflow so identical re-queues are cached, but any upstream change re-runs.
        if samples is None or vae is None or not hasattr(samples, "nodes") or not hasattr(vae, "ref"):
            return "uninitialized"
        key = json.dumps({
            "nodes": samples.nodes,
            "vae_ref": vae.ref,
            "samples_ref": samples.ref,
            "poll_interval": poll_interval, "timeout": timeout,
        }, sort_keys=True, default=str)
        return hashlib.sha256(key.encode()).hexdigest()

    def run(self, samples, vae, poll_interval, timeout):
        merged = merge_node_dicts(samples.nodes, vae.nodes)
        dec_id = fresh_id()
        merged[dec_id] = {"class_type": "VAEDecode",
                          "inputs": {"samples": samples.ref, "vae": vae.ref}}
        save_id = fresh_id()
        merged[save_id] = {"class_type": "SaveImage",
                           "inputs": {"images": [dec_id, 0], "filename_prefix": "cloud_decode"}}

        print(f"[CloudAPI] Assembled workflow with {len(merged)} nodes; submitting...")
        images, prompt_id = submit_and_collect_images(merged, poll_interval=poll_interval, timeout=timeout)
        return (images, prompt_id)
