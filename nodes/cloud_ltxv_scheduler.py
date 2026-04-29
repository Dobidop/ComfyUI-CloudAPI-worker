from ..handles import add_node, merge_node_dicts, fresh_id, CloudHandle


class CloudLTXVScheduler:
    """LTX-specific scheduler producing CLOUD_SIGMAS. Optional latent input
    informs the schedule (used by some LTX variants)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "steps":      ("INT",   {"default": 20, "min": 1, "max": 10000}),
                "max_shift":  ("FLOAT", {"default": 2.05, "min": 0.0, "max": 100.0, "step": 0.01}),
                "base_shift": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 100.0, "step": 0.01}),
                "stretch":    ("BOOLEAN", {"default": True}),
                "terminal":   ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.99, "step": 0.01}),
            },
            "optional": {
                "latent": ("CLOUD_LATENT",),
            },
        }

    RETURN_TYPES = ("CLOUD_SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    FUNCTION = "run"
    CATEGORY = "cloud"

    def run(self, steps, max_shift, base_shift, stretch, terminal, latent=None):
        merged = merge_node_dicts(*([latent.nodes] if latent else []))
        inputs = {
            "steps": steps, "max_shift": max_shift, "base_shift": base_shift,
            "stretch": stretch, "terminal": terminal,
        }
        if latent is not None:
            inputs["latent"] = latent.ref
        node_id = fresh_id()
        merged[node_id] = {"class_type": "LTXVScheduler", "inputs": inputs}
        return (CloudHandle(merged, [node_id, 0]),)
