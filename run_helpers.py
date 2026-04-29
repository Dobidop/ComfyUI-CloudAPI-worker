"""Shared submit→poll→download flow used by terminal cloud nodes."""
import json as _json

from .api_client import ComfyCloudClient
from .config import get_api_key, get_base_url
from .image_utils import bytes_to_tensor, stack_tensors


def _looks_like_image_entry(d):
    if not isinstance(d, dict):
        return False
    return ("filename" in d or "name" in d) and not any(
        k in d for k in ("class_type", "inputs", "node_errors")
    )


def extract_image_outputs(history):
    """Walk the history_v2 response and collect anything that looks like an image entry."""
    found = []

    def _walk(node):
        if isinstance(node, list):
            if node and all(_looks_like_image_entry(x) for x in node):
                found.extend(node)
                return
            for item in node:
                _walk(item)
        elif isinstance(node, dict):
            images = node.get("images")
            if isinstance(images, list) and all(_looks_like_image_entry(x) for x in images):
                found.extend(images)
            for k, v in node.items():
                if k == "images":
                    continue
                _walk(v)

    _walk(history)
    seen, unique = set(), []
    for entry in found:
        key = entry.get("filename") or entry.get("name")
        if key and key not in seen:
            seen.add(key)
            unique.append(entry)
    return unique


def submit_and_collect_images(workflow_dict, poll_interval=3.0, timeout=600):
    """Submit a workflow, poll until done, download all output images. Returns (image_tensor_batch, prompt_id)."""
    client = ComfyCloudClient(get_api_key(), get_base_url())
    prompt_id = client.submit_prompt(workflow_dict)
    print(f"[CloudAPI] Submitted prompt_id={prompt_id}")

    last_state = [None]
    def _log(state, _status):
        if state != last_state[0]:
            print(f"[CloudAPI] {prompt_id} -> {state}")
            last_state[0] = state

    final = client.poll_until_done(prompt_id, interval=poll_interval, timeout=timeout, on_status=_log)
    if final.get("status") not in client.SUCCESS_STATES:
        raise RuntimeError(
            f"Cloud job {prompt_id} ended with status='{final.get('status')}': "
            f"{final.get('error_message', '(no error message)')}"
        )

    history = client.get_history(prompt_id)
    outputs = extract_image_outputs(history)
    if not outputs:
        print(f"[CloudAPI] No outputs found. Raw history (truncated):")
        try:
            print(_json.dumps(history, indent=2, default=str)[:4000])
        except Exception:
            print(repr(history)[:4000])
        raise RuntimeError(f"No output images found in history for {prompt_id}.")

    tensors = []
    for entry in outputs:
        filename = entry.get("filename") or entry.get("name")
        if not filename:
            continue
        file_type = entry.get("type", "output")
        data = client.download_view(filename, file_type=file_type)
        tensors.append(bytes_to_tensor(data))

    if not tensors:
        raise RuntimeError(f"Could not download any output images for {prompt_id}")

    return stack_tensors(tensors), prompt_id


VIDEO_EXTENSIONS = (".mp4", ".webm", ".mov", ".mkv", ".avi", ".webp", ".gif")


def extract_video_outputs(history):
    """Find video file entries in history. The cloud's SaveVideo node puts MP4 entries
    into outputs[node_id].images alongside animated:[true]. We discriminate by file extension."""
    found = []

    def _walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if isinstance(v, list):
                    for entry in v:
                        if (isinstance(entry, dict)
                            and isinstance(entry.get("filename"), str)
                            and entry["filename"].lower().endswith(VIDEO_EXTENSIONS)):
                            found.append(entry)
                _walk(v)
        elif isinstance(node, list):
            for x in node:
                _walk(x)

    _walk(history)
    seen, unique = set(), []
    for entry in found:
        key = entry.get("filename")
        if key and key not in seen:
            seen.add(key)
            unique.append(entry)
    return unique


def submit_and_collect_video_files(workflow_dict, poll_interval=3.0, timeout=600):
    """Submit a workflow ending in SaveVideo (or anything producing a video file),
    return ([(filename, bytes), ...], prompt_id)."""
    client = ComfyCloudClient(get_api_key(), get_base_url())
    prompt_id = client.submit_prompt(workflow_dict)
    print(f"[CloudAPI] Submitted prompt_id={prompt_id} (video)")

    last_state = [None]
    def _log(state, _status):
        if state != last_state[0]:
            print(f"[CloudAPI] {prompt_id} -> {state}")
            last_state[0] = state

    final = client.poll_until_done(prompt_id, interval=poll_interval, timeout=timeout, on_status=_log)
    if final.get("status") not in client.SUCCESS_STATES:
        raise RuntimeError(
            f"Cloud job {prompt_id} ended with status='{final.get('status')}': "
            f"{final.get('error_message', '(no error message)')}"
        )

    history = client.get_history(prompt_id)
    entries = extract_video_outputs(history)
    if not entries:
        print(f"[CloudAPI] No video outputs found. Raw history (truncated):")
        try:
            print(_json.dumps(history, indent=2, default=str)[:4000])
        except Exception:
            print(repr(history)[:4000])
        raise RuntimeError(f"No video outputs found in history for {prompt_id}.")

    pairs = []
    for entry in entries:
        filename = entry["filename"]
        file_type = entry.get("type", "output")
        data = client.download_view(filename, file_type=file_type)
        pairs.append((filename, data))
    return pairs, prompt_id


def extract_latent_outputs(history):
    """Walk history_v2 response and return latent file entries (from SaveLatent nodes).
    The cloud returns these under outputs[<node_id>].latents = [{filename, subfolder, type}, ...]."""
    found = []

    def _walk(node):
        if isinstance(node, dict):
            latents = node.get("latents")
            if isinstance(latents, list):
                for entry in latents:
                    if isinstance(entry, dict) and (entry.get("filename") or entry.get("name")):
                        found.append(entry)
            for k, v in node.items():
                if k == "latents":
                    continue
                _walk(v)
        elif isinstance(node, list):
            for x in node:
                _walk(x)

    _walk(history)
    return found


def submit_and_collect_latent_files(workflow_dict, poll_interval=3.0, timeout=600):
    """Submit a workflow ending in SaveLatent, return ([latent_file_bytes, ...], prompt_id)."""
    client = ComfyCloudClient(get_api_key(), get_base_url())
    prompt_id = client.submit_prompt(workflow_dict)
    print(f"[CloudAPI] Submitted prompt_id={prompt_id} (latent materialize)")

    last_state = [None]
    def _log(state, _status):
        if state != last_state[0]:
            print(f"[CloudAPI] {prompt_id} -> {state}")
            last_state[0] = state

    final = client.poll_until_done(prompt_id, interval=poll_interval, timeout=timeout, on_status=_log)
    if final.get("status") not in client.SUCCESS_STATES:
        raise RuntimeError(
            f"Cloud job {prompt_id} ended with status='{final.get('status')}': "
            f"{final.get('error_message', '(no error message)')}"
        )

    history = client.get_history(prompt_id)
    entries = extract_latent_outputs(history)
    if not entries:
        print(f"[CloudAPI] No latent outputs found. Raw history (truncated):")
        try:
            print(_json.dumps(history, indent=2, default=str)[:4000])
        except Exception:
            print(repr(history)[:4000])
        raise RuntimeError(f"No latent outputs found in history for {prompt_id}.")

    blobs = []
    for entry in entries:
        filename = entry.get("filename") or entry.get("name")
        if not filename:
            continue
        file_type = entry.get("type", "output")
        blobs.append(client.download_view(filename, file_type=file_type))

    if not blobs:
        raise RuntimeError(f"Could not download any latent files for {prompt_id}")
    return blobs, prompt_id
