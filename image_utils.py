import io
import numpy as np
import torch
from PIL import Image


def tensor_to_pil(tensor):
    """ComfyUI IMAGE tensor (1, H, W, C) float32 [0,1] -> PIL.Image RGB."""
    if tensor.ndim == 4:
        tensor = tensor[0]
    arr = (tensor.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def pil_to_tensor(pil_image):
    """PIL.Image -> ComfyUI IMAGE tensor (1, H, W, C) float32 [0,1]."""
    img = pil_image.convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None,]


def bytes_to_tensor(image_bytes):
    return pil_to_tensor(Image.open(io.BytesIO(image_bytes)))


def stack_tensors(tensor_list):
    """Stack a list of (1,H,W,C) tensors into one (N,H,W,C) batch.
    If sizes differ, returns the first one only (cloud should produce uniform batches)."""
    if not tensor_list:
        return torch.zeros((1, 64, 64, 3), dtype=torch.float32)
    if len(tensor_list) == 1:
        return tensor_list[0]
    shapes = {t.shape for t in tensor_list}
    if len(shapes) == 1:
        return torch.cat(tensor_list, dim=0)
    return tensor_list[0]
