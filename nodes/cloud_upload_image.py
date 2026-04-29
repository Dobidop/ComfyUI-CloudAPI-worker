"""Bridges a local IMAGE into the cloud chain. Uploads on execute, returns
a CLOUD_IMAGE handle pointing at a LoadImage node that references the upload."""
from ..api_client import ComfyCloudClient
from ..config import get_api_key, get_base_url
from ..handles import CloudHandle, fresh_id
from ..image_utils import tensor_to_pil


class CloudUploadImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":    ("IMAGE",),
                "filename": ("STRING", {"default": "uploaded.png"}),
            }
        }

    RETURN_TYPES = ("CLOUD_IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "cloud"

    def run(self, image, filename):
        client = ComfyCloudClient(get_api_key(), get_base_url())
        cloud_filename = client.upload_image(tensor_to_pil(image), filename=filename)
        print(f"[CloudAPI] Uploaded image as '{cloud_filename}'")
        node_id = fresh_id()
        nodes = {node_id: {"class_type": "LoadImage", "inputs": {"image": cloud_filename}}}
        return (CloudHandle(nodes, [node_id, 0]),)
