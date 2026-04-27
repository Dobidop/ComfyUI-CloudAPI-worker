from ..config import get_cached_models
from ..handles import CloudHandle, fresh_id


def _ckpt_choices():
    cached = get_cached_models("checkpoints")
    return cached if cached else ["<run Cloud List Models (checkpoints) to populate>"]


class CloudCheckpointLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"ckpt_name": (_ckpt_choices(),)}}

    RETURN_TYPES = ("CLOUD_MODEL", "CLOUD_CLIP", "CLOUD_VAE")
    RETURN_NAMES = ("model", "clip", "vae")
    FUNCTION = "run"
    CATEGORY = "cloud"

    def run(self, ckpt_name):
        if ckpt_name.startswith("<"):
            raise RuntimeError("No checkpoint selected. Run Cloud List Models (folder=checkpoints) and restart ComfyUI.")
        nid = fresh_id()
        nodes = {nid: {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt_name}}}
        return (
            CloudHandle(nodes, [nid, 0]),
            CloudHandle(nodes, [nid, 1]),
            CloudHandle(nodes, [nid, 2]),
        )
