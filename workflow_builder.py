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

# Full sampler list as registered by KSamplerSelect on the cloud (Apr 2026).
# Bigger than the legacy SAMPLERS list (which we keep for backward compat with
# the existing CloudKSampler / CloudKSamplerAdvanced wrappers).
SAMPLERS_FULL = [
    "euler", "euler_cfg_pp", "euler_ancestral", "euler_ancestral_cfg_pp",
    "heun", "heunpp2", "exp_heun_2_x0", "exp_heun_2_x0_sde",
    "dpm_2", "dpm_2_ancestral", "lms", "dpm_fast", "dpm_adaptive",
    "dpmpp_2s_ancestral", "dpmpp_2s_ancestral_cfg_pp",
    "dpmpp_sde", "dpmpp_sde_gpu",
    "dpmpp_2m", "dpmpp_2m_cfg_pp", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu",
    "dpmpp_2m_sde_heun", "dpmpp_2m_sde_heun_gpu",
    "dpmpp_3m_sde", "dpmpp_3m_sde_gpu",
    "ddpm", "lcm", "ipndm", "ipndm_v", "deis",
    "res_multistep", "res_multistep_cfg_pp",
    "res_multistep_ancestral", "res_multistep_ancestral_cfg_pp",
    "gradient_estimation", "gradient_estimation_cfg_pp",
    "er_sde", "seeds_2", "seeds_3", "sa_solver", "sa_solver_pece",
    "ddim", "uni_pc", "uni_pc_bh2", "legacy_rk", "rk", "rk_beta",
    "deis_3m_ode", "deis_2m_ode", "deis_3m", "deis_2m",
    "res_6s_ode", "res_5s_ode", "res_3s_ode", "res_2s_ode",
    "res_3m_ode", "res_2m_ode",
    "res_6s", "res_5s", "res_3s", "res_2s", "res_3m", "res_2m",
]
