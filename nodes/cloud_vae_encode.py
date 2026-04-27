from ..api_client import ComfyCloudClient
from ..config import get_api_key, get_base_url
from ..handles import add_node, fresh_id, merge_node_dicts, CloudHandle
from ..image_utils import tensor_to_pil


class CloudVAEEncode:
    """Uploads the local image to the cloud, then appends LoadImage + VAEEncode
    to the assembled workflow. Output is a CLOUD_LATENT ref."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "vae":   ("CLOUD_VAE",),
            }
        }

    RETURN_TYPES = ("CLOUD_LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "run"
    CATEGORY = "cloud"

    def run(self, image, vae):
        client = ComfyCloudClient(get_api_key(), get_base_url())
        pil = tensor_to_pil(image)
        cloud_filename = client.upload_image(pil, filename="cloud_vae_encode_input.png")
        print(f"[CloudAPI] Uploaded image as '{cloud_filename}' for CloudVAEEncode")

        # Two new nodes: LoadImage feeds VAEEncode. Both share the assembled dict.
        merged = merge_node_dicts(vae.nodes)
        load_id = fresh_id()
        merged[load_id] = {"class_type": "LoadImage", "inputs": {"image": cloud_filename}}
        enc_id = fresh_id()
        merged[enc_id] = {"class_type": "VAEEncode",
                          "inputs": {"pixels": [load_id, 0], "vae": vae.ref}}
        return (CloudHandle(merged, [enc_id, 0]),)
