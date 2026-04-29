"""Cloud-side video assembly: CLOUD_IMAGE batch + fps -> CLOUD_VIDEO.

Optional CLOUD_AUDIO input (used by LTX2 audio workflows). Both fields wired
through to the cloud's CreateVideo node."""
from ..handles import CloudHandle, fresh_id, merge_node_dicts


class CloudCreateVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("CLOUD_IMAGE",),
                "fps":    ("FLOAT", {"default": 30.0, "min": 1.0, "max": 120.0, "step": 1.0}),
            },
            "optional": {
                "audio": ("CLOUD_AUDIO",),
            },
        }

    RETURN_TYPES = ("CLOUD_VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "run"
    CATEGORY = "cloud"

    def run(self, images, fps, audio=None):
        merged = merge_node_dicts(images.nodes, *([audio.nodes] if audio else []))
        inputs = {"images": images.ref, "fps": fps}
        if audio is not None:
            inputs["audio"] = audio.ref
        node_id = fresh_id()
        merged[node_id] = {"class_type": "CreateVideo", "inputs": inputs}
        return (CloudHandle(merged, [node_id, 0]),)
