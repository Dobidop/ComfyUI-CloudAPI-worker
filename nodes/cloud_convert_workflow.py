import json
import os
import re

import folder_paths

from ..converter import convert_workflow, conversion_summary


def _safe_filename(name):
    name = name.strip() or "converted_workflow"
    if not name.endswith(".json"):
        name += ".json"
    return re.sub(r'[<>:"/\\|?*]', "_", name)


class CloudConvertWorkflow:
    """Take a local ComfyUI workflow JSON and produce a Cloud variant.

    Accepts either:
      - **Editor format** (regular Save): top-level 'nodes' array. For workflows with
        no subgraphs this gives a fully-editable converted graph back, including positions.
      - **API format** (Save (API Format)): a flat {id: {class_type, inputs}} dict.
        Required for workflows that contain subgraphs (LTX 2.0 templates etc.) since
        the converter doesn't expand subgraphs itself — ComfyUI's frontend already
        flattens them when you save in API format.

    Boundary nodes (LoadImage, PreviewImage, VHS_VideoCombine, notes, etc.) are left alone.
    The converted workflow is written to ComfyUI's output directory; open it from there to use it.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "workflow_json":   ("STRING", {"multiline": True, "default": ""}),
                "output_filename": ("STRING", {"default": "converted_workflow.json"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("converted_json", "saved_path")
    FUNCTION = "run"
    CATEGORY = "cloud"
    OUTPUT_NODE = True

    def run(self, workflow_json, output_filename):
        if not workflow_json.strip():
            raise RuntimeError("workflow_json is empty. Paste the contents of a saved workflow .json.")

        try:
            workflow = json.loads(workflow_json)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Could not parse workflow JSON: {e}")

        converted, warnings = convert_workflow(workflow)

        out_dir = folder_paths.get_output_directory()
        out_path = os.path.join(out_dir, _safe_filename(output_filename))
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(converted, f, indent=2, ensure_ascii=False)

        summary = conversion_summary(workflow, converted)
        print(f"[CloudAPI] {summary} Saved to: {out_path}")
        if warnings:
            print(f"[CloudAPI] {len(warnings)} warning(s):")
            for w in warnings:
                print(f"  - {w}")

        result_text = json.dumps(converted, indent=2, ensure_ascii=False)
        return (result_text, out_path)
