from .startup import kick_off_prefetch
from .nodes.cloud_ksampler           import CloudKSampler
from .nodes.cloud_list_models        import CloudListModels
from .nodes.cloud_checkpoint_loader  import CloudCheckpointLoader
from .nodes.cloud_lora_loader        import CloudLoraLoader
from .nodes.cloud_clip_text_encode   import CloudCLIPTextEncode
from .nodes.cloud_empty_latent       import CloudEmptyLatent
from .nodes.cloud_vae_encode         import CloudVAEEncode
from .nodes.cloud_ksampler_graph     import CloudKSamplerGraph
from .nodes.cloud_vae_decode         import CloudVAEDecode
from .nodes.cloud_unet_loader        import CloudUNETLoader
from .nodes.cloud_vae_loader         import CloudVAELoader
from .nodes.cloud_clip_loader        import CloudCLIPLoader
from .nodes.cloud_dual_clip_loader   import CloudDualCLIPLoader
from .nodes.cloud_empty_latent_video import CloudEmptyLatentVideo
from .nodes.cloud_lora_loader_model_only import CloudLoraLoaderModelOnly
from .nodes.cloud_model_sampling_sd3   import CloudModelSamplingSD3
from .nodes.cloud_ksampler_advanced    import CloudKSamplerAdvanced
from .nodes.cloud_wan_image_to_video   import CloudWanImageToVideo
from .nodes.cloud_convert_workflow     import CloudConvertWorkflow
from .nodes.cloud_materialize_latent   import CloudMaterializeLatent

NODE_CLASS_MAPPINGS = {
    "CloudKSampler":          CloudKSampler,
    "CloudListModels":        CloudListModels,
    "CloudCheckpointLoader":  CloudCheckpointLoader,
    "CloudLoraLoader":        CloudLoraLoader,
    "CloudCLIPTextEncode":    CloudCLIPTextEncode,
    "CloudEmptyLatent":       CloudEmptyLatent,
    "CloudVAEEncode":         CloudVAEEncode,
    "CloudKSamplerGraph":     CloudKSamplerGraph,
    "CloudVAEDecode":         CloudVAEDecode,
    "CloudUNETLoader":        CloudUNETLoader,
    "CloudVAELoader":         CloudVAELoader,
    "CloudCLIPLoader":        CloudCLIPLoader,
    "CloudDualCLIPLoader":    CloudDualCLIPLoader,
    "CloudEmptyLatentVideo":  CloudEmptyLatentVideo,
    "CloudLoraLoaderModelOnly": CloudLoraLoaderModelOnly,
    "CloudModelSamplingSD3":  CloudModelSamplingSD3,
    "CloudKSamplerAdvanced":  CloudKSamplerAdvanced,
    "CloudWanImageToVideo":   CloudWanImageToVideo,
    "CloudConvertWorkflow":   CloudConvertWorkflow,
    "CloudMaterializeLatent": CloudMaterializeLatent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CloudKSampler":          "Cloud KSampler (Quick)",
    "CloudListModels":        "Cloud List Models",
    "CloudCheckpointLoader":  "Cloud Checkpoint Loader",
    "CloudLoraLoader":        "Cloud Lora Loader",
    "CloudCLIPTextEncode":    "Cloud CLIP Text Encode",
    "CloudEmptyLatent":       "Cloud Empty Latent",
    "CloudVAEEncode":         "Cloud VAE Encode",
    "CloudKSamplerGraph":     "Cloud KSampler",
    "CloudVAEDecode":         "Cloud VAE Decode",
    "CloudUNETLoader":        "Cloud UNET Loader",
    "CloudVAELoader":         "Cloud VAE Loader",
    "CloudCLIPLoader":        "Cloud CLIP Loader",
    "CloudDualCLIPLoader":    "Cloud Dual CLIP Loader",
    "CloudEmptyLatentVideo":  "Cloud Empty Latent Video",
    "CloudLoraLoaderModelOnly": "Cloud Lora Loader (Model Only)",
    "CloudModelSamplingSD3":  "Cloud Model Sampling SD3",
    "CloudKSamplerAdvanced":  "Cloud KSampler Advanced",
    "CloudWanImageToVideo":   "Cloud Wan Image To Video",
    "CloudConvertWorkflow":   "Cloud Convert Workflow",
    "CloudMaterializeLatent": "Cloud Materialize Latent",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

# Background prefetch of model lists. Daemon thread — never blocks ComfyUI startup;
# logs and skips on missing key / network errors.
kick_off_prefetch()
