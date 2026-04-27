"""Builders that emit ComfyUI workflow JSON dicts for the Cloud API."""


def build_text2img(ckpt, positive, negative, width, height, batch_size,
                   seed, steps, cfg, sampler_name, scheduler, denoise):
    return {
        "1": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": ckpt}},
        "2": {"class_type": "CLIPTextEncode",
              "inputs": {"text": positive, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode",
              "inputs": {"text": negative, "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage",
              "inputs": {"width": width, "height": height, "batch_size": batch_size}},
        "5": {"class_type": "KSampler",
              "inputs": {
                  "seed": seed, "steps": steps, "cfg": cfg,
                  "sampler_name": sampler_name, "scheduler": scheduler, "denoise": denoise,
                  "model": ["1", 0], "positive": ["2", 0],
                  "negative": ["3", 0], "latent_image": ["4", 0],
              }},
        "6": {"class_type": "VAEDecode",
              "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage",
              "inputs": {"images": ["6", 0], "filename_prefix": "cloud_t2i"}},
    }


def build_img2img(ckpt, positive, negative, cloud_image_filename,
                  seed, steps, cfg, sampler_name, scheduler, denoise):
    return {
        "1": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": ckpt}},
        "2": {"class_type": "CLIPTextEncode",
              "inputs": {"text": positive, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode",
              "inputs": {"text": negative, "clip": ["1", 1]}},
        "4": {"class_type": "LoadImage",
              "inputs": {"image": cloud_image_filename}},
        "5": {"class_type": "VAEEncode",
              "inputs": {"pixels": ["4", 0], "vae": ["1", 2]}},
        "6": {"class_type": "KSampler",
              "inputs": {
                  "seed": seed, "steps": steps, "cfg": cfg,
                  "sampler_name": sampler_name, "scheduler": scheduler, "denoise": denoise,
                  "model": ["1", 0], "positive": ["2", 0],
                  "negative": ["3", 0], "latent_image": ["5", 0],
              }},
        "7": {"class_type": "VAEDecode",
              "inputs": {"samples": ["6", 0], "vae": ["1", 2]}},
        "8": {"class_type": "SaveImage",
              "inputs": {"images": ["7", 0], "filename_prefix": "cloud_i2i"}},
    }


SAMPLERS = [
    "euler", "euler_ancestral", "heun", "heunpp2", "dpm_2", "dpm_2_ancestral",
    "lms", "dpm_fast", "dpm_adaptive",
    "dpmpp_2s_ancestral", "dpmpp_sde", "dpmpp_sde_gpu",
    "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu",
    "dpmpp_3m_sde", "dpmpp_3m_sde_gpu",
    "ddpm", "lcm", "ddim", "uni_pc", "uni_pc_bh2",
]

SCHEDULERS = [
    "normal", "karras", "exponential", "sgm_uniform",
    "simple", "ddim_uniform", "beta",
]

MODEL_FOLDERS = [
    "checkpoints", "loras", "vae", "embeddings",
    "controlnet", "upscale_models", "clip", "clip_vision",
    "diffusion_models", "unet", "style_models", "hypernetworks",
    "text_encoders",
]

UNET_WEIGHT_DTYPES = [
    "default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2",
]

CLIP_TYPES = [
    "stable_diffusion", "stable_cascade", "sd3", "stable_audio",
    "mochi", "ltxv", "pixart", "cosmos", "lumina2", "wan",
    "hidream", "chroma", "ace", "omnigen2",
]

DUAL_CLIP_TYPES = [
    "sdxl", "sd3", "flux", "hunyuan_video", "hidream",
]

VIDEO_LATENT_FORMATS = [
    "hunyuan_wan",  # EmptyHunyuanLatentVideo (also works for Wan)
    "ltxv",         # EmptyLTXVLatentVideo
]
