import re, dgl, torch
from typing import List, Dict

# keep only these seven labels
KEEP_LABELS = {
    # "Background",
    # "Cause",
    # "Contrast",
    "Elaboration",
    # "Evaluation",
    # "Explanation",
    "Joint",
}
BRIDGE = "rst-bridge"

# full 18-label list in the ORIGINAL index order
FULL_LABELS = [
    "Attribution","Background","Cause","Comparison","Condition","Contrast",
    "Elaboration","Enablement","Evaluation","Explanation","Joint",
    "Manner-Means","Same-Unit","Summary","Temporal","TextualOrganization",
    "Topic-Change","Topic-Comment"
]

class RSTGraphProcessor:
    """Token-level RST graph with only the KEEP_LABELS (+ bridge)."""

    _node_pat = re.compile(r"\((\d+):[^=]+=([A-Za-z\-]+):\d+,\s*(\d+):")

    def __init__(self, rst_list: List[dict]):
        self.rst_list = rst_list

        # rel2id uses the original string keys that rgat.py will look up
        self.rel2id = {f"rst-{lbl}": i for i, lbl in enumerate(sorted(KEEP_LABELS))}
        self.rel2id[BRIDGE] = len(self.rel2id)
        self.id2rel = {i: r for r, i in self.rel2id.items()}

    # ───── helpers ────────────────────────────────────────────────────
    @staticmethod
    def split_edus(cuts: List[int], tokens: List[str]):
        edus, start = [], 0
        for cut in cuts:
            edus.append(list(range(start, cut)))
            start = cut
        return edus

    @staticmethod
    def pick_rel(logits: List[float]) -> str:
        return FULL_LABELS[int(max(range(len(logits)), key=logits.__getitem__))]

    # ───── main builder ───────────────────────────────────────────────
    def build_graph(self, sample_id: int, tokens: List[str]) -> Dict:
        js   = self.rst_list[sample_id]
        edus = self.split_edus(js["all_segmentation_pred"], tokens)
        edu2tok = {i: tok_ids for i, tok_ids in enumerate(edus, start=1)}

        edges: List[tuple] = []
        for tree_str, logits in zip(js["all_tree_parsing_pred"],
                                    js["all_relation_logits"]):
            for head, rel_txt, dep in self._node_pat.findall(tree_str):
                rel = self.pick_rel(logits) if rel_txt == "span" else rel_txt
                if rel not in KEEP_LABELS:
                    continue                                   # skip unwanted
                rel_key = f"rst-{rel}"
                h, d = int(head), int(dep)
                for s in edu2tok[h]:
                    for t in edu2tok[d]:
                        edges.append((s, t, rel_key))

        # always add bridge
        edges.append((0, len(tokens) - 1, BRIDGE))

        src = torch.tensor([e[0] for e in edges], dtype=torch.int32)
        dst = torch.tensor([e[1] for e in edges], dtype=torch.int32)
        graph = dgl.graph((src, dst), num_nodes=len(tokens), idtype=torch.int32)

        return {
            "graph": graph,
            "edges": edges   # ← list of (src,dst,relation_string)
        }
