# ComfyUI-CloudAPI-worker

Proof-of-concept ComfyUI custom node pack that runs workflows on the [ComfyUI Cloud](https://cloud.comfy.org) from inside a local ComfyUI graph. You wire up familiar-looking nodes (Cloud Checkpoint Loader, Cloud KSampler, Cloud VAE Decode, etc.) and they assemble a workflow JSON, ship it to the cloud, poll for completion, and return images as a normal `IMAGE` tensor.

## How it works

The cloud nodes don't run anything locally. Each one returns a lightweight handle (`CLOUD_MODEL`, `CLOUD_CLIP`, `CLOUD_VAE`, `CLOUD_CONDITIONING`, `CLOUD_LATENT`) that carries the partial workflow JSON accumulated so far plus a reference to a node-output slot. Only the terminal node — `Cloud VAE Decode` — actually submits to the cloud, polls, and downloads the results. Whatever shape you wire up locally translates 1:1 into the JSON sent to the cloud, so LoRAs, separate CLIP loaders, dual-text-encoder setups, two-stage samplers, etc. all just work.

Models, LoRAs, VAEs, and text encoders all live cloud-side and never traverse the wire. Local images (e.g. for img2img or Wan I2V) are uploaded once via `/api/upload/image` and referenced as a `LoadImage` node in the assembled workflow.

## Installation

1. Clone or copy this folder into `ComfyUI/custom_nodes/`.
2. Copy `config.json.example` to `config.json` and paste your API key from <https://platform.comfy.org/profile/api-keys>.
3. Restart ComfyUI. Model dropdowns populate automatically in the background — checkpoints, loras, vae, diffusion_models, text_encoders, clip_vision are prefetched on startup.

## Nodes

All nodes appear under the **cloud** category.

**Loaders:** Cloud Checkpoint Loader, Cloud UNET Loader, Cloud VAE Loader, Cloud CLIP Loader, Cloud Dual CLIP Loader, Cloud Lora Loader, Cloud Lora Loader (Model Only)

**Conditioning:** Cloud CLIP Text Encode, Cloud Model Sampling SD3, Cloud Wan Image To Video

**Latents:** Cloud Empty Latent, Cloud Empty Latent Video, Cloud VAE Encode

**Sampling:** Cloud KSampler, Cloud KSampler Advanced, Cloud KSampler (Quick) *(monolithic txt2img/img2img all-in-one)*

**Terminal:** Cloud VAE Decode *(submits assembled workflow, downloads images)*

**Bridge:** Cloud Materialize Latent *(runs the cloud chain so far, downloads the latent as a local LATENT — useful for inserting a local-only operation like a 3rd-party latent upscaler. One-way: comfy-cloud has no LoadLatent support, so a local LATENT can't be pushed back into a cloud chain.)*

**Utility:** Cloud List Models *(refreshes a folder's dropdown cache on demand),* Cloud Convert Workflow *(translates an editor-format local workflow into its cloud equivalent and writes it to ComfyUI's output dir)*

## Notes

- Polling status updates print to the console (`queued_waiting → allocated → preparing → executing → success`).
- Bump the `timeout` on Cloud VAE Decode for long jobs. Wan 2.2 14B I2V with the 4-step LoRA finishes in ~70-100s; without the LoRA it can take 8-10 minutes.
- For video workflows, the IMAGE batch returned by Cloud VAE Decode is N frames — wire it into VHS_VideoCombine to mux locally.
- The dropdown cache lives in `config.json` and refreshes every 24h on startup. Run a Cloud List Models node to force-refresh a specific folder, then restart ComfyUI.

## Status

Proof of concept. Architecture is solid for image and video diffusion workflows; missing pieces include ControlNet, CLIP Vision conditioning, dedicated cloud-side video encoding (`SaveVideo`/`SaveAnimatedWebP`), and per-model I2V conditioners other than Wan. Add as needed.

## License

MIT — see [LICENSE](LICENSE).
