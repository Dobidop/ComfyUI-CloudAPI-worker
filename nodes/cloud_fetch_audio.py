"""Terminal: CLOUD_AUDIO -> AUDIO.

Appends a SaveAudio (FLAC) to the assembled cloud workflow, submits, polls,
downloads the audio file, and returns it as a ComfyUI AUDIO dict
({"waveform": (1, channels, samples) tensor, "sample_rate": int})."""
import hashlib
import io
import json

from ..handles import fresh_id, merge_node_dicts
from ..run_helpers import submit_and_collect_audio_files


def _bytes_to_audio(audio_bytes):
    """Decode FLAC/WAV/MP3/etc bytes into ComfyUI's AUDIO dict format."""
    import torchaudio
    waveform, sample_rate = torchaudio.load(io.BytesIO(audio_bytes))
    if waveform.ndim == 2:
        waveform = waveform.unsqueeze(0)  # (channels, samples) -> (1, channels, samples)
    return {"waveform": waveform, "sample_rate": int(sample_rate)}


class CloudFetchAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio":           ("CLOUD_AUDIO",),
                "filename_prefix": ("STRING", {"default": "cloud_fetch_audio"}),
                "poll_interval":   ("FLOAT", {"default": 3.0, "min": 0.5, "max": 60.0, "step": 0.5}),
                "timeout":         ("INT",   {"default": 1800, "min": 10, "max": 7200}),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "prompt_id")
    FUNCTION = "run"
    CATEGORY = "cloud"

    @classmethod
    def IS_CHANGED(cls, audio=None, filename_prefix="cloud_fetch_audio",
                   poll_interval=3.0, timeout=1800):
        if audio is None or not hasattr(audio, "nodes"):
            return "uninitialized"
        key = json.dumps({
            "nodes": audio.nodes, "ref": audio.ref,
            "filename_prefix": filename_prefix,
            "poll_interval": poll_interval, "timeout": timeout,
        }, sort_keys=True, default=str)
        return hashlib.sha256(key.encode()).hexdigest()

    def run(self, audio, filename_prefix, poll_interval, timeout):
        merged = merge_node_dicts(audio.nodes)
        save_id = fresh_id()
        merged[save_id] = {
            "class_type": "SaveAudio",
            "inputs": {"audio": audio.ref, "filename_prefix": filename_prefix},
        }
        print(f"[CloudAPI] Fetching audio: assembled {len(merged)} nodes; submitting...")
        pairs, prompt_id = submit_and_collect_audio_files(
            merged, poll_interval=poll_interval, timeout=timeout
        )
        if len(pairs) > 1:
            print(f"[CloudAPI] {len(pairs)} audio files returned; using the first.")
        _, data = pairs[0]
        audio_dict = _bytes_to_audio(data)
        print(f"[CloudAPI] Decoded audio: shape={tuple(audio_dict['waveform'].shape)} "
              f"sample_rate={audio_dict['sample_rate']}")
        return (audio_dict, prompt_id)
