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
from .nodes.cloud_vae_decode_ref       import CloudVAEDecodeRef
from .nodes.cloud_create_video         import CloudCreateVideo
from .nodes.cloud_save_video           import CloudSaveVideo
from .nodes.cloud_cfg_guider           import CloudCFGGuider
from .nodes.cloud_ksampler_select      import CloudKSamplerSelect
from .nodes.cloud_random_noise         import CloudRandomNoise
from .nodes.cloud_manual_sigmas        import CloudManualSigmas
from .nodes.cloud_sampler_custom_advanced import CloudSamplerCustomAdvanced
from .nodes.cloud_empty_image          import CloudEmptyImage
from .nodes.cloud_get_image_size       import CloudGetImageSize
from .nodes.cloud_image_scale_by       import CloudImageScaleBy
from .nodes.cloud_resize_images_by_longer_edge import CloudResizeImagesByLongerEdge
from .nodes.cloud_ltxv_conditioning    import CloudLTXVConditioning
from .nodes.cloud_ltxv_scheduler       import CloudLTXVScheduler
from .nodes.cloud_ltxv_preprocess      import CloudLTXVPreprocess
from .nodes.cloud_ltxv_img_to_video_inplace import CloudLTXVImgToVideoInplace
from .nodes.cloud_ltxv_crop_guides     import CloudLTXVCropGuides
from .nodes.cloud_ltxv_latent_upsampler import CloudLTXVLatentUpsampler
from .nodes.cloud_ltxav_text_encoder_loader import CloudLTXAVTextEncoderLoader
from .nodes.cloud_ltxv_audio_vae_loader import CloudLTXVAudioVAELoader
from .nodes.cloud_ltxv_audio_vae_decode import CloudLTXVAudioVAEDecode
from .nodes.cloud_ltxv_concat_av_latent import CloudLTXVConcatAVLatent
from .nodes.cloud_ltxv_separate_av_latent import CloudLTXVSeparateAVLatent
from .nodes.cloud_ltxv_empty_latent_audio import CloudLTXVEmptyLatentAudio
from .nodes.cloud_latent_upscale_model_loader import CloudLatentUpscaleModelLoader

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
    "CloudVAEDecodeRef":      CloudVAEDecodeRef,
    "CloudCreateVideo":       CloudCreateVideo,
    "CloudSaveVideo":         CloudSaveVideo,
    "CloudCFGGuider":         CloudCFGGuider,
    "CloudKSamplerSelect":    CloudKSamplerSelect,
    "CloudRandomNoise":       CloudRandomNoise,
    "CloudManualSigmas":      CloudManualSigmas,
    "CloudSamplerCustomAdvanced": CloudSamplerCustomAdvanced,
    "CloudEmptyImage":        CloudEmptyImage,
    "CloudGetImageSize":      CloudGetImageSize,
    "CloudImageScaleBy":      CloudImageScaleBy,
    "CloudResizeImagesByLongerEdge": CloudResizeImagesByLongerEdge,
    "CloudLTXVConditioning":  CloudLTXVConditioning,
    "CloudLTXVScheduler":     CloudLTXVScheduler,
    "CloudLTXVPreprocess":    CloudLTXVPreprocess,
    "CloudLTXVImgToVideoInplace": CloudLTXVImgToVideoInplace,
    "CloudLTXVCropGuides":    CloudLTXVCropGuides,
    "CloudLTXVLatentUpsampler": CloudLTXVLatentUpsampler,
    "CloudLTXAVTextEncoderLoader": CloudLTXAVTextEncoderLoader,
    "CloudLTXVAudioVAELoader": CloudLTXVAudioVAELoader,
    "CloudLTXVAudioVAEDecode": CloudLTXVAudioVAEDecode,
    "CloudLTXVConcatAVLatent": CloudLTXVConcatAVLatent,
    "CloudLTXVSeparateAVLatent": CloudLTXVSeparateAVLatent,
    "CloudLTXVEmptyLatentAudio": CloudLTXVEmptyLatentAudio,
    "CloudLatentUpscaleModelLoader": CloudLatentUpscaleModelLoader,
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
    "CloudVAEDecodeRef":      "Cloud VAE Decode (Ref)",
    "CloudCreateVideo":       "Cloud Create Video",
    "CloudSaveVideo":         "Cloud Save Video",
    "CloudCFGGuider":         "Cloud CFG Guider",
    "CloudKSamplerSelect":    "Cloud KSampler Select",
    "CloudRandomNoise":       "Cloud Random Noise",
    "CloudManualSigmas":      "Cloud Manual Sigmas",
    "CloudSamplerCustomAdvanced": "Cloud Sampler Custom Advanced",
    "CloudEmptyImage":        "Cloud Empty Image",
    "CloudGetImageSize":      "Cloud Get Image Size",
    "CloudImageScaleBy":      "Cloud Image Scale By",
    "CloudResizeImagesByLongerEdge": "Cloud Resize Images By Longer Edge",
    "CloudLTXVConditioning":  "Cloud LTXV Conditioning",
    "CloudLTXVScheduler":     "Cloud LTXV Scheduler",
    "CloudLTXVPreprocess":    "Cloud LTXV Preprocess",
    "CloudLTXVImgToVideoInplace": "Cloud LTXV Img To Video Inplace",
    "CloudLTXVCropGuides":    "Cloud LTXV Crop Guides",
    "CloudLTXVLatentUpsampler": "Cloud LTXV Latent Upsampler",
    "CloudLTXAVTextEncoderLoader": "Cloud LTXAV Text Encoder Loader",
    "CloudLTXVAudioVAELoader": "Cloud LTXV Audio VAE Loader",
    "CloudLTXVAudioVAEDecode": "Cloud LTXV Audio VAE Decode",
    "CloudLTXVConcatAVLatent": "Cloud LTXV Concat AV Latent",
    "CloudLTXVSeparateAVLatent": "Cloud LTXV Separate AV Latent",
    "CloudLTXVEmptyLatentAudio": "Cloud LTXV Empty Latent Audio",
    "CloudLatentUpscaleModelLoader": "Cloud Latent Upscale Model Loader",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

# Background prefetch of model lists. Daemon thread — never blocks ComfyUI startup;
# logs and skips on missing key / network errors.
kick_off_prefetch()
