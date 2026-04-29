from ..handles import add_node
from ..workflow_builder import SAMPLERS_FULL


class CloudKSamplerSelect:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampler_name": (SAMPLERS_FULL,),
            }
        }

    RETURN_TYPES = ("CLOUD_SAMPLER",)
    RETURN_NAMES = ("sampler",)
    FUNCTION = "run"
    CATEGORY = "cloud"

    def run(self, sampler_name):
        (handle,) = add_node([], "KSamplerSelect", {"sampler_name": sampler_name})
        return (handle,)
