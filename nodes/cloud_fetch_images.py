"""Terminal: CLOUD_IMAGE -> IMAGE batch.

Appends a SaveImage to the assembled cloud workflow, submits, polls, downloads
all output PNGs, and returns them as a single IMAGE tensor batch."""
import hashlib
import json

from ..handles import fresh_id, merge_node_dicts
from ..run_helpers import submit_and_collect_images


class CloudFetchImages:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images":          ("CLOUD_IMAGE",),
                "filename_prefix": ("STRING", {"default": "cloud_fetch"}),
                "poll_interval":   ("FLOAT", {"default": 3.0, "min": 0.5, "max": 60.0, "step": 0.5}),
                "timeout":         ("INT",   {"default": 1800, "min": 10, "max": 7200}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "prompt_id")
    FUNCTION = "run"
    CATEGORY = "cloud"

    @classmethod
    def IS_CHANGED(cls, images=None, filename_prefix="cloud_fetch",
                   poll_interval=3.0, timeout=1800):
        if images is None or not hasattr(images, "nodes"):
            return "uninitialized"
        key = json.dumps({
            "nodes": images.nodes, "ref": images.ref,
            "filename_prefix": filename_prefix,
            "poll_interval": poll_interval, "timeout": timeout,
        }, sort_keys=True, default=str)
        return hashlib.sha256(key.encode()).hexdigest()

    def run(self, images, filename_prefix, poll_interval, timeout):
        merged = merge_node_dicts(images.nodes)
        save_id = fresh_id()
        merged[save_id] = {
            "class_type": "SaveImage",
            "inputs": {"images": images.ref, "filename_prefix": filename_prefix},
        }
        print(f"[CloudAPI] Fetching images: assembled {len(merged)} nodes; submitting...")
        tensor, prompt_id = submit_and_collect_images(
            merged, poll_interval=poll_interval, timeout=timeout
        )
        return (tensor, prompt_id)
