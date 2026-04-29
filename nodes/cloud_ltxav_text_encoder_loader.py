from ..handles import add_node


class CloudLTXAVTextEncoderLoader:
    """LTX 2.0 AV text encoder loader. The cloud's options list is huge and
    drifts as new encoders are added — we accept any STRING and let the cloud
    validate. Copy the value from your local LTX workflow's widget to be sure."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_encoder": ("STRING", {"default": "umt5_xxl_fp8_e4m3fn_scaled.safetensors"}),
                "ckpt_name": ("STRING", {"default": "ltx-2-19b-dev-fp8.safetensors"}),
                "device": ("STRING", {"default": "default"}),
            }
        }

    RETURN_TYPES = ("CLOUD_CLIP",)
    RETURN_NAMES = ("CLIP",)
    FUNCTION = "run"
    CATEGORY = "cloud"

    def run(self, text_encoder, ckpt_name, device):
        (handle,) = add_node([], "LTXAVTextEncoderLoader", {"text_encoder": text_encoder, "ckpt_name": ckpt_name, "device": device})
        return (handle,)
