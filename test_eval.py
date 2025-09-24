#!/usr/bin/env python
# eval_anl.py
# usage:  python eval_anl.py formatted_output.txt

import sys
import re
from pathlib import Path

from discourse_graph.utils import decode_anl_v2
from discourse_graph.evaluate import BatchEvaluator   # ← adjust import

# ----------------------------------------------------------------------
# helper: pull (true, pred) lines out of the file
# ----------------------------------------------------------------------
def load_pairs(txt_path):
    true_line, pred_line = None, None
    for raw in Path(txt_path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("True:"):
            true_line = line[len("True:"):].strip()
        elif line.startswith("Pred:"):
            pred_line = line[len("Pred:"):].strip()
            if true_line is not None:
                yield true_line, pred_line
                true_line, pred_line = None, None
    # silently ignores any unmatched trailing “True:” block

# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main(txt_file):
    be = BatchEvaluator()

    for true_txt, pred_txt in load_pairs(txt_file):
        true_comp, true_rel = decode_anl_v2(true_txt)
        pred_comp, pred_rel = decode_anl_v2(pred_txt)

        # BatchEvaluator expects lists of samples; wrap each sample in a list
        be.add_batch([true_comp], [true_rel], [pred_comp], [pred_rel])

    scores = be.evaluate()

    print("\n=== evaluation ===")
    for k, v in scores.items():
        print(f"{k:12s}: {v:.4f}")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python eval_anl.py <file_with_true_pred_blocks.txt>")
        sys.exit(1)
    main(sys.argv[1])
