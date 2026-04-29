"""Convert a local ComfyUI editor-format workflow into a Cloud variant.

The conversion is mostly mechanical: rename the node `type`, rename socket types
(MODEL → CLOUD_MODEL etc.), and apply per-node-type widget transforms to account
for slight shape differences (e.g. dropping the UI-only `control_after_generate`
widget, prepending a `format` choice for the unified video latent node, appending
`poll_interval` / `timeout` to the terminal VAE Decode).

Any node not in LOCAL_TO_CLOUD is left untouched, which keeps boundary nodes
(LoadImage, PreviewImage, SaveImage, VHS_VideoCombine, notes, etc.) intact.
The IMAGE socket type intentionally does NOT get renamed — that's the bridge
between local and cloud subgraphs (Cloud VAE Decode → PreviewImage,
LoadImage → Cloud VAE Encode, etc.).
"""
import copy


LOCAL_TO_CLOUD = {
    "CheckpointLoaderSimple":   "CloudCheckpointLoader",
    "UNETLoader":               "CloudUNETLoader",
    "VAELoader":                "CloudVAELoader",
    "CLIPLoader":               "CloudCLIPLoader",
    "DualCLIPLoader":           "CloudDualCLIPLoader",
    "LoraLoader":               "CloudLoraLoader",
    "LoraLoaderModelOnly":      "CloudLoraLoaderModelOnly",
    "CLIPTextEncode":           "CloudCLIPTextEncode",
    "EmptyLatentImage":         "CloudEmptyLatent",
    "EmptyHunyuanLatentVideo":  "CloudEmptyLatentVideo",
    "EmptyLTXVLatentVideo":     "CloudEmptyLatentVideo",
    "KSampler":                 "CloudKSamplerGraph",
    "KSamplerAdvanced":         "CloudKSamplerAdvanced",
    "ModelSamplingSD3":         "CloudModelSamplingSD3",
    "WanImageToVideo":          "CloudWanImageToVideo",
    "VAEEncode":                "CloudVAEEncode",
    "VAEDecode":                "CloudVAEDecode",
}

# Local socket types that have a cloud counterpart. IMAGE is intentionally absent —
# it remains the boundary type between local and cloud subgraphs.
TYPE_MAP = {
    "MODEL":        "CLOUD_MODEL",
    "CLIP":         "CLOUD_CLIP",
    "VAE":          "CLOUD_VAE",
    "CONDITIONING": "CLOUD_CONDITIONING",
    "LATENT":       "CLOUD_LATENT",
}


# --- Per-node widget transforms ----------------------------------------------
# Each takes the node's `widgets_values` list and returns the transformed list
# matching the cloud node's widget order.

# Note on seed widgets: ComfyUI's frontend auto-inserts a `control_after_generate`
# entry into widgets_values for any INT field named "seed"/"noise_seed" with a large
# max. This applies to both the local nodes AND our cloud nodes (same INT shape), so
# their widgets_values arrays line up identically — no transform needed for KSampler
# or KSamplerAdvanced.

def _empty_hunyuan_video(values):
    # Cloud unified node prepends a `format` widget choosing the latent layout.
    return ["hunyuan_wan"] + list(values)


def _empty_ltxv_video(values):
    return ["ltxv"] + list(values)


def _vae_decode(values):
    # CloudVAEDecode is terminal; appends poll_interval + timeout widgets.
    return list(values) + [3.0, 600]


WIDGET_TRANSFORMS = {
    "EmptyHunyuanLatentVideo":  _empty_hunyuan_video,
    "EmptyLTXVLatentVideo":     _empty_ltxv_video,
    "VAEDecode":                _vae_decode,
}


def convert_workflow(workflow):
    """Convert a deep copy of `workflow` (editor format) and return (converted_dict, warnings)."""
    workflow = copy.deepcopy(workflow)
    warnings = []

    nodes = workflow.get("nodes") or []
    links = workflow.get("links") or []

    converted_node_ids = set()

    # Pass 1: rewrite each convertible node's type, socket types, widgets, properties
    for node in nodes:
        local_type = node.get("type")
        if local_type not in LOCAL_TO_CLOUD:
            continue

        cloud_type = LOCAL_TO_CLOUD[local_type]
        node["type"] = cloud_type
        converted_node_ids.add(node.get("id"))

        if local_type in WIDGET_TRANSFORMS:
            node["widgets_values"] = WIDGET_TRANSFORMS[local_type](node.get("widgets_values") or [])

        for slot in node.get("inputs") or []:
            t = slot.get("type")
            if t in TYPE_MAP:
                slot["type"] = TYPE_MAP[t]
        for slot in node.get("outputs") or []:
            t = slot.get("type")
            if t in TYPE_MAP:
                slot["type"] = TYPE_MAP[t]

        props = node.get("properties")
        if isinstance(props, dict):
            props["Node name for S&R"] = cloud_type
            # These reference comfy-core; the node is now a custom node.
            props.pop("cnr_id", None)
            props.pop("ver", None)
            # The "models" hint pinned to comfy-core folders no longer applies.
            props.pop("models", None)

    # Pass 2: rewrite link types; warn on broken boundary connections
    for link in links:
        # Editor link entries are arrays: [link_id, src_node, src_slot, dst_node, dst_slot, "TYPE"]
        if not isinstance(link, list) or len(link) < 6:
            continue
        link_id, src_id, _src_slot, dst_id, _dst_slot, link_type = link[:6]
        if link_type not in TYPE_MAP:
            continue

        src_converted = src_id in converted_node_ids
        dst_converted = dst_id in converted_node_ids
        cloud_type = TYPE_MAP[link_type]

        if src_converted and dst_converted:
            link[5] = cloud_type
        elif src_converted ^ dst_converted:
            unconverted = dst_id if src_converted else src_id
            warnings.append(
                f"Link {link_id} ({link_type}): bridges converted and unconverted nodes "
                f"(node {unconverted} not converted). The connection will likely break — "
                "either convert that node too or break the link manually."
            )

    return workflow, warnings


def conversion_summary(original, converted):
    """Return a one-line summary string."""
    n_total = len(original.get("nodes") or [])
    n_converted = sum(
        1 for n in (converted.get("nodes") or []) if n.get("type", "").startswith("Cloud")
    )
    return f"Converted {n_converted}/{n_total} nodes."
