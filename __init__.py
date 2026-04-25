from .nodes.cloud_ksampler import CloudKSampler
from .nodes.cloud_list_models import CloudListModels

NODE_CLASS_MAPPINGS = {
    "CloudKSampler":   CloudKSampler,
    "CloudListModels": CloudListModels,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CloudKSampler":   "Cloud KSampler",
    "CloudListModels": "Cloud List Models",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
