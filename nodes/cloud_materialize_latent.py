"""Bridge node: pulls a CLOUD_LATENT down to a local LATENT.

Appends a SaveLatent to the assembled cloud workflow, submits, polls, downloads
the resulting `.latent` file, and deserializes it locally — mirroring comfy-core's
LoadLatent semantics. Useful for running a local-only operation (e.g. a 3rd-party
latent upscaler) midway through an otherwise-cloud pipeline.

One-way only. Comfy-cloud has no LoadLatent support, so a CloudAdoptLatent does NOT
exist — anything downstream of this node must stay local."""
import hashlib
import json
import os
import tempfile

import safetensors.torch

from ..handles import fresh_id, merge_node_dicts
from ..run_helpers import submit_and_collect_latent_files


def _bytes_to_latent(latent_bytes):
    """Mirror comfy.core LoadLatent.load(): deserialize safetensors blob, optionally rescale."""
    try:
        sd = safetensors.torch.load(latent_bytes)
    except Exception:
        # Older safetensors versions lack .load(bytes); fall back to tempfile + best-effort cleanup.
        # On Windows the mmap may keep the file locked after load_file returns, so we can't
        # always unlink immediately — leave it for the OS tempdir cleaner if so.
        tmp = tempfile.NamedTemporaryFile(suffix=".latent", delete=False)
        try:
            tmp.write(latent_bytes)
            tmp.close()
            sd = safetensors.torch.load_file(tmp.name)
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass
    multiplier = 1.0 if "latent_format_version_0" in sd else 1.0 / 0.18215
    return {"samples": sd["latent_tensor"].float() * multiplier}


class CloudMaterializeLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples":       ("CLOUD_LATENT",),
                "poll_interval": ("FLOAT", {"default": 3.0, "min": 0.5, "max": 60.0, "step": 0.5}),
                "timeout":       ("INT",   {"default": 600, "min": 10, "max": 3600}),
            }
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("samples", "prompt_id")
    FUNCTION = "run"
    CATEGORY = "cloud"

    @classmethod
    def IS_CHANGED(cls, samples=None, poll_interval=3.0, timeout=600):
        if samples is None or not hasattr(samples, "nodes"):
            return "uninitialized"
        key = json.dumps(
            {"nodes": samples.nodes, "ref": samples.ref,
             "poll_interval": poll_interval, "timeout": timeout},
            sort_keys=True, default=str,
        )
        return hashlib.sha256(key.encode()).hexdigest()

    def run(self, samples, poll_interval, timeout):
        merged = merge_node_dicts(samples.nodes)
        save_id = fresh_id()
        merged[save_id] = {
            "class_type": "SaveLatent",
            "inputs": {"samples": samples.ref, "filename_prefix": "cloud_materialize"},
        }
        print(f"[CloudAPI] Materializing latent: assembled {len(merged)} nodes; submitting...")
        blobs, prompt_id = submit_and_collect_latent_files(
            merged, poll_interval=poll_interval, timeout=timeout
        )
        if len(blobs) > 1:
            print(f"[CloudAPI] {len(blobs)} latent files returned; using the first.")
        latent = _bytes_to_latent(blobs[0])
        print(f"[CloudAPI] Materialized latent shape: {tuple(latent['samples'].shape)}")
        return (latent, prompt_id)
