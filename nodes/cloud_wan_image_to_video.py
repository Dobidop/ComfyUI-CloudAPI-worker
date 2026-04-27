from ..api_client import ComfyCloudClient
from ..config import get_api_key, get_base_url
from ..handles import CloudHandle, fresh_id, merge_node_dicts
from ..image_utils import tensor_to_pil


class CloudWanImageToVideo:
    """Wan 2.x image-to-video conditioning. Encodes a start image into the first frames of
    the latent and biases conditioning to continue from it.

    The optional start_image is uploaded automatically; clip_vision_output is unused by
    default Wan 2.2 I2V (link is null in the official template) but accepted for completeness."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive":   ("CLOUD_CONDITIONING",),
                "negative":   ("CLOUD_CONDITIONING",),
                "vae":        ("CLOUD_VAE",),
                "width":      ("INT", {"default": 640, "min": 64, "max": 8192, "step": 8}),
                "height":     ("INT", {"default": 640, "min": 64, "max": 8192, "step": 8}),
                "length":     ("INT", {"default": 81,  "min": 1,  "max": 4096}),
                "batch_size": ("INT", {"default": 1,   "min": 1,  "max": 16}),
            },
            "optional": {
                "start_image":         ("IMAGE",),
                "clip_vision_output":  ("CLOUD_CLIP_VISION_OUTPUT",),
            },
        }

    RETURN_TYPES = ("CLOUD_CONDITIONING", "CLOUD_CONDITIONING", "CLOUD_LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "run"
    CATEGORY = "cloud"

    def run(self, positive, negative, vae, width, height, length, batch_size,
            start_image=None, clip_vision_output=None):

        merged = merge_node_dicts(positive.nodes, negative.nodes, vae.nodes,
                                  *( [clip_vision_output.nodes] if clip_vision_output else [] ))

        inputs = {
            "positive": positive.ref,
            "negative": negative.ref,
            "vae":      vae.ref,
            "width":    width,
            "height":   height,
            "length":   length,
            "batch_size": batch_size,
        }

        if start_image is not None:
            client = ComfyCloudClient(get_api_key(), get_base_url())
            cloud_filename = client.upload_image(tensor_to_pil(start_image), filename="wan_i2v_start.png")
            print(f"[CloudAPI] Uploaded Wan I2V start image as '{cloud_filename}'")
            load_id = fresh_id()
            merged[load_id] = {"class_type": "LoadImage", "inputs": {"image": cloud_filename}}
            inputs["start_image"] = [load_id, 0]

        if clip_vision_output is not None:
            inputs["clip_vision_output"] = clip_vision_output.ref

        node_id = fresh_id()
        merged[node_id] = {"class_type": "WanImageToVideo", "inputs": inputs}
        return (
            CloudHandle(merged, [node_id, 0]),
            CloudHandle(merged, [node_id, 1]),
            CloudHandle(merged, [node_id, 2]),
        )
