"""Terminal: CLOUD_VIDEO -> VIDEO.

Like CloudSaveVideo, but instead of just saving to disk and returning a path,
this returns a ComfyUI VIDEO object that can be wired into further local
processing (e.g. an upscale pass, frame extraction, re-encoding). Also writes
the MP4 to ComfyUI's output dir as a side-effect, so the VIDEO is backed by a
real file."""
import hashlib
import json
import os
import time

import folder_paths

from ..handles import fresh_id, merge_node_dicts
from ..run_helpers import submit_and_collect_video_files


def _make_video_object(local_path):
    """Wrap a local MP4 path as a ComfyUI VIDEO. The VideoFromFile import path has
    moved between versions; try a couple of locations before giving up."""
    try:
        from comfy_api.input_impl import VideoFromFile  # current ComfyUI 0.3.x
        return VideoFromFile(local_path)
    except Exception:
        pass
    try:
        from comfy_api.input_impl.video_types import VideoFromFile  # alternate path
        return VideoFromFile(local_path)
    except Exception:
        pass
    raise RuntimeError(
        "Could not locate ComfyUI's VideoFromFile class to wrap the downloaded MP4. "
        "Use Cloud Save Video instead — it returns the local file path."
    )


class CloudFetchVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video":           ("CLOUD_VIDEO",),
                "filename_prefix": ("STRING", {"default": "cloud_fetch_video"}),
                "format":          (["auto", "mp4"], {"default": "auto"}),
                "codec":           (["auto", "h264"], {"default": "auto"}),
                "poll_interval":   ("FLOAT", {"default": 3.0, "min": 0.5, "max": 60.0, "step": 0.5}),
                "timeout":         ("INT",   {"default": 1800, "min": 10, "max": 7200}),
            }
        }

    RETURN_TYPES = ("VIDEO", "STRING", "STRING")
    RETURN_NAMES = ("video", "file_path", "prompt_id")
    FUNCTION = "run"
    CATEGORY = "cloud"

    @classmethod
    def IS_CHANGED(cls, video=None, filename_prefix="cloud_fetch_video",
                   format="auto", codec="auto", poll_interval=3.0, timeout=1800):
        if video is None or not hasattr(video, "nodes"):
            return "uninitialized"
        key = json.dumps({
            "nodes": video.nodes, "ref": video.ref,
            "filename_prefix": filename_prefix,
            "format": format, "codec": codec,
            "poll_interval": poll_interval, "timeout": timeout,
        }, sort_keys=True, default=str)
        return hashlib.sha256(key.encode()).hexdigest()

    def run(self, video, filename_prefix, format, codec, poll_interval, timeout):
        merged = merge_node_dicts(video.nodes)
        save_id = fresh_id()
        merged[save_id] = {
            "class_type": "SaveVideo",
            "inputs": {
                "video":           video.ref,
                "filename_prefix": filename_prefix,
                "format":          format,
                "codec":           codec,
            },
        }
        print(f"[CloudAPI] Fetching video: assembled {len(merged)} nodes; submitting...")
        pairs, prompt_id = submit_and_collect_video_files(
            merged, poll_interval=poll_interval, timeout=timeout
        )
        if len(pairs) > 1:
            print(f"[CloudAPI] {len(pairs)} video files returned; using the first.")

        cloud_filename, data = pairs[0]
        ext = os.path.splitext(cloud_filename)[1] or ".mp4"
        ts = time.strftime("%Y%m%d-%H%M%S")
        local_filename = f"{filename_prefix}_{ts}{ext}"
        out_dir = folder_paths.get_output_directory()
        local_path = os.path.join(out_dir, local_filename)
        with open(local_path, "wb") as f:
            f.write(data)
        print(f"[CloudAPI] Wrote {len(data)} bytes -> {local_path}")

        video_obj = _make_video_object(local_path)
        return (video_obj, local_path, prompt_id)
