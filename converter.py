"""Convert a local ComfyUI workflow into a Cloud variant.

Supports two formats:
  - **Editor format** (default Save): `{nodes: [...], links: [...], definitions: {...}}`
    Mostly mechanical class_type rename + socket-type rename. Subgraphs are NOT expanded
    here — workflows containing them must be exported in API format first.
  - **API format** (Save (API Format)): a flat `{node_id: {class_type, inputs}, ...}`.
    ComfyUI's frontend already flattens subgraphs into this form, so this is the
    recommended format for any workflow with subgraphs (e.g. LTX 2.0 templates).

Boundary nodes (LoadImage, PreviewImage, SaveImage, VHS_VideoCombine, notes, etc.)
are left untouched. The IMAGE socket type intentionally never gets renamed — that's
the bridge between local and cloud subgraphs.
"""
import copy


# Local class_type -> Cloud class_type. Keep alphabetized within groups for readability.
LOCAL_TO_CLOUD = {
    # Loaders
    "CheckpointLoaderSimple":   "CloudCheckpointLoader",
    "UNETLoader":               "CloudUNETLoader",
    "VAELoader":                "CloudVAELoader",
    "CLIPLoader":               "CloudCLIPLoader",
    "DualCLIPLoader":           "CloudDualCLIPLoader",
    "LoraLoader":               "CloudLoraLoader",
    "LoraLoaderModelOnly":      "CloudLoraLoaderModelOnly",
    "LatentUpscaleModelLoader": "CloudLatentUpscaleModelLoader",

    # Conditioning / sampling primitives
    "CLIPTextEncode":           "CloudCLIPTextEncode",
    "ModelSamplingSD3":         "CloudModelSamplingSD3",
    "CFGGuider":                "CloudCFGGuider",
    "KSamplerSelect":           "CloudKSamplerSelect",
    "RandomNoise":              "CloudRandomNoise",
    "ManualSigmas":             "CloudManualSigmas",

    # Latents
    "EmptyLatentImage":         "CloudEmptyLatent",
    "EmptyHunyuanLatentVideo":  "CloudEmptyLatentVideo",
    "EmptyLTXVLatentVideo":     "CloudEmptyLatentVideo",

    # Samplers
    "KSampler":                 "CloudKSamplerGraph",
    "KSamplerAdvanced":         "CloudKSamplerAdvanced",
    "SamplerCustomAdvanced":    "CloudSamplerCustomAdvanced",

    # Wan
    "WanImageToVideo":          "CloudWanImageToVideo",

    # LTX video
    "LTXVConditioning":         "CloudLTXVConditioning",
    "LTXVScheduler":            "CloudLTXVScheduler",
    "LTXVPreprocess":           "CloudLTXVPreprocess",
    "LTXVImgToVideoInplace":    "CloudLTXVImgToVideoInplace",
    "LTXVCropGuides":           "CloudLTXVCropGuides",
    "LTXVLatentUpsampler":      "CloudLTXVLatentUpsampler",

    # LTX audio
    "LTXAVTextEncoderLoader":   "CloudLTXAVTextEncoderLoader",
    "LTXVAudioVAELoader":       "CloudLTXVAudioVAELoader",
    "LTXVAudioVAEDecode":       "CloudLTXVAudioVAEDecode",
    "LTXVConcatAVLatent":       "CloudLTXVConcatAVLatent",
    "LTXVSeparateAVLatent":     "CloudLTXVSeparateAVLatent",
    "LTXVEmptyLatentAudio":     "CloudLTXVEmptyLatentAudio",

    # Image utilities
    "EmptyImage":               "CloudEmptyImage",
    "GetImageSize":             "CloudGetImageSize",
    "ImageScaleBy":             "CloudImageScaleBy",
    "ResizeImagesByLongerEdge": "CloudResizeImagesByLongerEdge",

    # VAE encode/decode (decode is non-terminal; fetch nodes auto-inserted at boundaries)
    "VAEEncode":                "CloudVAEEncode",
    "VAEDecode":                "CloudVAEDecode",

    # Video out
    "CreateVideo":              "CloudCreateVideo",
    "SaveVideo":                "CloudSaveVideo",
}

# Local socket types that have a cloud counterpart. IMAGE intentionally absent —
# it remains the boundary type between local and cloud subgraphs.
TYPE_MAP = {
    "MODEL":                "CLOUD_MODEL",
    "CLIP":                 "CLOUD_CLIP",
    "VAE":                  "CLOUD_VAE",
    "CONDITIONING":         "CLOUD_CONDITIONING",
    "LATENT":               "CLOUD_LATENT",
    "GUIDER":               "CLOUD_GUIDER",
    "SAMPLER":              "CLOUD_SAMPLER",
    "SIGMAS":               "CLOUD_SIGMAS",
    "NOISE":                "CLOUD_NOISE",
    "LATENT_UPSCALE_MODEL": "CLOUD_LATENT_UPSCALE_MODEL",
    "AUDIO":                "CLOUD_AUDIO",
    "VIDEO":                "CLOUD_VIDEO",
}


# --- Format detection ----------------------------------------------------------

def _is_editor_format(wf):
    return isinstance(wf, dict) and "nodes" in wf and isinstance(wf.get("nodes"), list)


def _has_subgraphs(wf):
    defs = wf.get("definitions") or {}
    return bool(defs.get("subgraphs"))


def _is_api_format(wf):
    """Heuristic: a flat dict where every value is a node-spec object with class_type."""
    if not isinstance(wf, dict):
        return False
    if not wf:
        return False
    return all(
        isinstance(v, dict) and "class_type" in v
        for v in wf.values()
    )


# --- Cloud node output type table (used by fetch auto-insertion) --------------
# Slot index -> CLOUD_X type. Only nodes whose outputs may flow into local
# (unconverted) consumers via a fetch bridge need to be listed here.
CLOUD_NODE_OUTPUT_TYPES = {
    "CloudVAEDecode":         {0: "CLOUD_IMAGE"},
    "CloudCreateVideo":       {0: "CLOUD_VIDEO"},
    "CloudLTXVAudioVAEDecode": {0: "CLOUD_AUDIO"},
}

# CLOUD_X -> (fetch class_type, source-input-name on the fetch node)
CLOUD_TO_FETCH = {
    "CLOUD_IMAGE": ("CloudFetchImages", "images"),
    "CLOUD_AUDIO": ("CloudFetchAudio", "audio"),
    "CLOUD_VIDEO": ("CloudFetchVideo", "video"),
}


# --- Editor-format conversion (kept from previous version) ---------------------

def _empty_hunyuan_video_widgets(values):
    return ["hunyuan_wan"] + list(values)


def _empty_ltxv_video_widgets(values):
    return ["ltxv"] + list(values)


WIDGET_TRANSFORMS = {
    "EmptyHunyuanLatentVideo":  _empty_hunyuan_video_widgets,
    "EmptyLTXVLatentVideo":     _empty_ltxv_video_widgets,
}


def _convert_editor_format(workflow):
    workflow = copy.deepcopy(workflow)
    warnings = []

    nodes = workflow.get("nodes") or []
    links = workflow.get("links") or []
    converted_node_ids = set()

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
            props.pop("cnr_id", None)
            props.pop("ver", None)
            props.pop("models", None)

    for link in links:
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


# --- API-format conversion -----------------------------------------------------

def _api_inline_primitive_int(workflow):
    """Find PrimitiveInt nodes and inline their value into all consumer inputs."""
    primitives = {nid: node for nid, node in workflow.items()
                  if node.get("class_type") == "PrimitiveInt"}
    if not primitives:
        return 0
    inlined = 0
    for nid, pnode in primitives.items():
        value = (pnode.get("inputs") or {}).get("value", 0)
        for cid, cnode in workflow.items():
            if cid == nid:
                continue
            inputs = cnode.get("inputs") or {}
            for k, v in list(inputs.items()):
                if isinstance(v, list) and len(v) == 2 and v[0] == nid:
                    inputs[k] = value
                    inlined += 1
    for nid in list(primitives.keys()):
        del workflow[nid]
    return inlined


def _api_insert_fetch_nodes(workflow):
    """For each link from a converted (cloud) node to an unconverted (local) node,
    insert a fetch terminal between them when the source's output is a known
    CLOUD_X with an associated fetch node. Returns count of insertions."""
    converted = {nid for nid, n in workflow.items()
                 if (n.get("class_type") or "").startswith("Cloud")}
    inserted = 0
    new_nodes = {}
    for cid, node in list(workflow.items()):
        if cid in converted:
            continue
        inputs = node.get("inputs") or {}
        for k, v in list(inputs.items()):
            if not (isinstance(v, list) and len(v) == 2):
                continue
            src_id, src_slot = v
            if src_id not in converted:
                continue
            src_class = workflow[src_id].get("class_type")
            cloud_out_type = CLOUD_NODE_OUTPUT_TYPES.get(src_class, {}).get(src_slot)
            if not cloud_out_type:
                continue
            if cloud_out_type not in CLOUD_TO_FETCH:
                continue
            fetch_class, fetch_input_name = CLOUD_TO_FETCH[cloud_out_type]
            fetch_id = f"_auto_fetch_{src_id}_{src_slot}"
            # Reuse the same fetch node if multiple consumers share the same source slot
            if fetch_id not in workflow and fetch_id not in new_nodes:
                new_nodes[fetch_id] = {
                    "class_type": fetch_class,
                    "inputs": {
                        fetch_input_name: [src_id, src_slot],
                        "filename_prefix": "cloud_fetch",
                        "poll_interval": 3.0,
                        "timeout": 1800,
                    },
                }
            inputs[k] = [fetch_id, 0]
            inserted += 1
    workflow.update(new_nodes)
    return inserted


def _convert_api_format(workflow):
    workflow = copy.deepcopy(workflow)
    warnings = []

    inlined = _api_inline_primitive_int(workflow)
    if inlined:
        warnings.append(f"Inlined {inlined} PrimitiveInt reference(s) into consumer widgets.")

    # Pass 1: rename class_types and adjust inputs as needed
    for nid, node in workflow.items():
        local_type = node.get("class_type")
        if local_type in LOCAL_TO_CLOUD:
            node["class_type"] = LOCAL_TO_CLOUD[local_type]
            inputs = node.setdefault("inputs", {})

            # Special widget-injection cases for nodes whose cloud variant has extra fields
            if local_type == "EmptyHunyuanLatentVideo":
                inputs["format"] = "hunyuan_wan"
            elif local_type == "EmptyLTXVLatentVideo":
                inputs["format"] = "ltxv"
            elif local_type == "SaveVideo":
                inputs.setdefault("poll_interval", 3.0)
                inputs.setdefault("timeout", 1800)

    # Pass 2: auto-insert fetch terminals at cloud->local boundaries
    inserted = _api_insert_fetch_nodes(workflow)
    if inserted:
        warnings.append(
            f"Inserted {inserted} fetch terminal(s) (Cloud Fetch Images / Cloud Fetch Audio) "
            "where converted cloud outputs feed local nodes."
        )

    return workflow, warnings


# --- Public entry point --------------------------------------------------------

def convert_workflow(workflow):
    """Detect format and dispatch. Returns (converted_dict, warnings)."""
    if _is_editor_format(workflow):
        if _has_subgraphs(workflow):
            raise RuntimeError(
                "This workflow contains subgraphs which the converter doesn't expand. "
                "Open the workflow in ComfyUI and use Workflow > Save (API Format) "
                "to flatten it first, then paste that API-format JSON instead."
            )
        return _convert_editor_format(workflow)
    if _is_api_format(workflow):
        return _convert_api_format(workflow)
    raise RuntimeError(
        "Could not recognize the workflow format. Expected either editor format "
        "(top-level 'nodes' array) or API format (flat dict of {id: {class_type, inputs}})."
    )


def conversion_summary(original, converted):
    """Return a one-line summary string."""
    if _is_editor_format(original):
        n_total = len(original.get("nodes") or [])
        n_converted = sum(
            1 for n in (converted.get("nodes") or []) if n.get("type", "").startswith("Cloud")
        )
    else:
        n_total = len(original)
        n_converted = sum(1 for v in converted.values() if v.get("class_type", "").startswith("Cloud"))
    return f"Converted {n_converted}/{n_total} nodes."
