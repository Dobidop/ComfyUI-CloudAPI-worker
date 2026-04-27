"""Cloud handle types — the lightweight refs that flow between graph-style cloud nodes.

A CloudHandle is NOT tensor data. It's a pair:
  - nodes: the partial workflow JSON dict accumulated so far
  - ref:   ["node_id", output_slot] pointing into that dict

Each cloud node merges its input handles' `nodes` dicts, appends one new node spec,
and returns one CloudHandle per output slot. The terminal node (CloudVAEDecode /
CloudSaveImage) submits the fully-assembled `nodes` dict to /api/prompt.
"""
import uuid


class CloudHandle:
    __slots__ = ("nodes", "ref")

    def __init__(self, nodes, ref):
        self.nodes = nodes
        self.ref = list(ref)

    def __repr__(self):
        return f"CloudHandle(ref={self.ref}, n_nodes={len(self.nodes)})"


def fresh_id():
    return "n_" + uuid.uuid4().hex[:10]


def merge_node_dicts(*dicts):
    """Union dicts. UUID-based IDs guarantee no key collisions between independent chains."""
    out = {}
    for d in dicts:
        out.update(d)
    return out


def add_node(input_handles, class_type, inputs, num_outputs=1):
    """Build one new workflow node referencing the given input handles.

    Returns a tuple of CloudHandle, one per output slot.
    All returned handles share the same `nodes` dict.
    """
    merged = merge_node_dicts(*[h.nodes for h in input_handles])
    new_id = fresh_id()
    merged[new_id] = {"class_type": class_type, "inputs": inputs}
    return tuple(CloudHandle(merged, [new_id, i]) for i in range(num_outputs))


def ref_of(handle):
    return list(handle.ref)
