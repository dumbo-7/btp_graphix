import re, pickle, dgl, torch
from collections import defaultdict
from typing import List, Dict

RST_LABELS = [
"Attribution",
"Background",
"Cause",
"Comparison",
"Condition",
"Contrast",
"Elaboration",
"Enablement",
"Evaluation",
"Explanation",
"Joint",
"Manner-Means",
"Same-Unit",
"Summary",
"Temporal",
"TextualOrganization",
"Topic-Change",
"Topic-Comment"
]

BRIDGE = "rst-bridge"

class RSTGraphProcessor:
    def __init__(self, rst_dict: Dict[int, dict]):
        self.rst_dict = rst_dict
        self.rel2id = {f"rst-{l}": i for i, l in enumerate(RST_LABELS)}
        self.rel2id[BRIDGE] = len(self.rel2id)          # add bridge
        self.id2rel = {i:r for r,i in self.rel2id.items()}

    # ------------------------------------------------------------------
    # helpers -----------------------------------------------------------
    # ------------------------------------------------------------------
    @staticmethod
    def split_edus(cuts: List[int], tokens: List[str]):
        """returns list(list(token_idx)) mapping edu_id → token indices"""
        edus, start = [], 0
        for cut in cuts:
            edus.append(list(range(start, cut)))
            start = cut
        return edus

    @staticmethod
    def pick_rel(rel_logits: List[float]) -> str:
        return RST_LABELS[int(max(range(len(rel_logits)), key=rel_logits.__getitem__))]

    _node_pat = re.compile(r'\((\d+):[^=]+=([A-Za-z\-]+):\d+,\s*(\d+):')

    def build_graph(self, sample_id: int, tokens: List[str]) -> dict:
        js   = self.rst_dict[sample_id]
        edus = self.split_edus(js["all_segmentation_pred"], tokens)

        # 1. map EDU idx → its token indices (for Cartesian edge creation)
        edu2tok = {i: tok_ids for i, tok_ids in enumerate(edus, start=1)}

        # 2. extract RST arcs from bracket string(s)
        edges = []
        for tree_str, logits in zip(js["all_tree_parsing_pred"], js["all_relation_logits"]):
            for head, rel_txt, dep in self._node_pat.findall(tree_str):
                # Resolve 'span' using logits
                rel = self.pick_rel(logits) if rel_txt == "span" else rel_txt
                rel = f"rst-{rel}"  # Prefix all relations for consistency
                h, d = int(head), int(dep)
                for s in edu2tok[h]:
                    for t in edu2tok[d]:
                        edges.append((s, t, rel))

        # 3. Add one bridge edge to keep the graph connected
        bridge_edge = (0, len(tokens) - 1, "rst-bridge")
        edges.append(bridge_edge)

        # 4. Create the DGL graph from src and dst
        src = [e[0] for e in edges]
        dst = [e[1] for e in edges]
        graph = dgl.graph((src, dst), num_nodes=len(tokens), idtype=torch.int32)

        return {
            "graph": graph,
            "edges": edges,  # List of (src, dst, relation_str)
            # "question_mask": [1] * len(tokens),  # All 1s since it's all text
            # "schema_mask": [1] * len(tokens)     # All 0s (no schema here)
        }


