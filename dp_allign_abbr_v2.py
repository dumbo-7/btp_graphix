#!/usr/bin/env python
# dp_align.py
# Align predicted ANL strings to reference strings (span‐only) and evaluate.

import re
from pathlib import Path

# ───────────────────────────────────────────────────────────────────────
# local imports – change the paths / names if your modules differ
# ───────────────────────────────────────────────────────────────────────
from discourse_graph.evaluate import BatchEvaluator   # unchanged evaluator
from discourse_graph.utils import decode_anl_v2       # returns ([(type, span)], [relations])
# ───────────────────────────────────────────────────────────────────────


# ----------------------------------------------------------------------
# tokenisation helpers
# ----------------------------------------------------------------------
def tokenize(txt: str):
    """Lower‐case, whitespace‐split tokens."""
    return txt.lower().strip().split()


def word_distance(a_tokens, b_tokens):
    """
    Bag‐of‐words Hamming + length diff on token lists.
    """
    return sum(x != y for x, y in zip(a_tokens, b_tokens)) + abs(len(a_tokens) - len(b_tokens))


# ----------------------------------------------------------------------
# bracket span parsing / reconstruction
# ----------------------------------------------------------------------
SPAN_RE = re.compile(r"\[\s*([^\[\]]+?)\s*\]")

def extract_spans(text: str):
    """
    Return a list of (full_span_str, start_idx, end_idx), e.g.:
    ("[ … | Premise | P2 | Supports = C1 ]", 123, 189)
    """
    return [(m.group(0), m.start(), m.end()) for m in SPAN_RE.finditer(text)]


def parse_span(span_text: str):
    """
    Split a bracketed component string into:
      main  : str   – free text (field 0)
      label : str   – Claim / Premise / …  (field 1, may be wrong)
      uid   : str   – C1 / P3 / …            (field 2, model‐generated)
      tail  : list  – any additional fields (relations, etc.)
    """
    parts = [p.strip() for p in span_text.strip("[]").split("|")]
    main  = parts[0] if len(parts) > 0 else ""
    label = parts[1] if len(parts) > 1 else ""
    uid   = parts[2] if len(parts) > 2 else ""
    tail  = parts[3:] if len(parts) > 3 else []
    return main, label, uid, tail


def build_span(main: str, label: str, uid: str, tail: list):
    """
    Reconstruct a bracket string from its parts:
    “[ main | label | uid | …tail ]”
    """
    return "[ " + " | ".join([main, label, uid] + tail) + " ]"


# ----------------------------------------------------------------------
# alignment core: match spans by text only (ignore types/UIDs/relations)
# ----------------------------------------------------------------------
def align_pred_to_true(true: str, pred: str, max_diff: int) -> str:
    """
    For every gold bracket G_i in `true`:
      1. Parse G_i → (g_main, g_label, g_uid, g_tail).
      2. Among *all* predicted brackets P_j in `pred`, find the one whose
         p_main is nearest to g_main by token_distance (≤ max_diff).
         (No check on p_label or p_uid or g_label.)
      3. Rebuild that predicted bracket as:
           [  g_main  | p_label  | p_uid  | p_tail...  ]
         i.e. keep the model’s label/UID/tail, but force the main text to
         be the gold text `g_main`.
      4. Record a replacement from the *entire* original P_j string to this
         new bracket string.
    5. After collecting all such replacements, apply them in order of descending
       length of the “old” P_j (to avoid partial‐substring overlaps).
    6. Return the fully reconstructed, “span‐aligned” prediction paragraph.
    """
    gold_spans = extract_spans(true)   # list of (g_str, g_start, g_end)
    pred_spans = extract_spans(pred)   # list of (p_str, p_start, p_end)

    replacements = []  # list of (old_pred_str, new_pred_str)
    used_preds    = set()  # to avoid aligning multiple golds to same pred bracket

    # 1) For each gold bracket, find the best predicted bracket to align
    for g_str, *_ in gold_spans:
        g_main, _, _, _ = parse_span(g_str)
        g_tok = tokenize(g_main)

        best_dist = None
        best_p_str = None
        best_parts = None

        for p_str, *_ in pred_spans:
            if p_str in used_preds:
                continue

            p_main, p_label, p_uid, p_tail = parse_span(p_str)
            dist = word_distance(g_tok, tokenize(p_main))

            if dist <= max_diff and (best_dist is None or dist < best_dist):
                best_dist  = dist
                best_p_str = p_str
                best_parts = (p_label, p_uid, p_tail)

        if best_p_str is None:
            continue  # no predicted bracket close enough to this gold span

        # We will align best_p_str’s main text to the gold text g_main
        p_label, p_uid, p_tail = best_parts
        new_pred_str = build_span(g_main, p_label, p_uid, p_tail)

        if new_pred_str != best_p_str:
            replacements.append((best_p_str, new_pred_str))
        used_preds.add(best_p_str)

    # 2) Apply replacements in order of decreasing length of old_pred_str
    replacements.sort(key=lambda tup: -len(tup[0]))
    aligned = pred
    for old_str, new_str in replacements:
        aligned = aligned.replace(old_str, new_str, 1)

    return aligned


# ----------------------------------------------------------------------
# file‐level processing: align each Pred paragraph to its True paragraph
# ----------------------------------------------------------------------
PAIR_RE = re.compile(r"True:(.*?)\nPred:(.*?)\n-+", re.DOTALL)

def process_file(file_path: str, max_diff: int) -> str:
    """
    Read <file_path> containing multiple blocks of:

      True: <some text with brackets>
      Pred: <some text with brackets>
      --------------------------------

    For each pair, run align_pred_to_true(...) on the Pred text,
    then re‐emit:

      True: <true_text>
      Pred: <aligned_pred_text>
      --------------------------------
    """
    raw = Path(file_path).read_text(encoding="utf-8")
    output = []

    for true_par, pred_par in PAIR_RE.findall(raw):
        aligned_pred = align_pred_to_true(true_par.strip(),
                                          pred_par.strip(),
                                          max_diff)
        output.append(
            f"True:{true_par.strip()}\n"
            f"Pred:{aligned_pred.strip()}\n"
            f"{'-'*80}"
        )

    return "\n".join(output)


# ----------------------------------------------------------------------
# evaluation: after writing aligned file, decode & compare tuples
# ----------------------------------------------------------------------
def evaluate_from_aligned_text(filepath: str):
    """
    Run BatchEvaluator on the aligned file:
      1. decode_anl_v2(true_par) → ([(type, span)], [relations])
      2. decode_anl_v2(pred_par) → ([(type, span)], [relations])
      3. feed aligned tuples into BatchEvaluator
    """
    evaluator = BatchEvaluator()
    text = Path(filepath).read_text(encoding="utf-8")

    for true_par, pred_par in PAIR_RE.findall(text):
        t_comps, t_rels = decode_anl_v2(true_par.strip())
        p_comps, p_rels = decode_anl_v2(pred_par.strip())

        evaluator.add_batch(
            [t_comps], [t_rels],
            [p_comps], [p_rels]
        )

    return evaluator.evaluate()


# ----------------------------------------------------------------------
# script entry‐point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    SRC_FILE   = "test.txt"                   # your raw True/Pred pairs
    ALIGNED_FN = "aligned_predictions.txt"    # will hold span‐aligned output
    MAX_DIFF   = 3                            # ≤2 token edits to align

    print("• Aligning predictions …")
    aligned_txt = process_file(SRC_FILE, max_diff=MAX_DIFF)
    Path(ALIGNED_FN).write_text(aligned_txt, encoding="utf-8")
    print(f"  ↳ aligned file saved to: {ALIGNED_FN}")

    print("\n• Evaluating …")
    metrics = evaluate_from_aligned_text(ALIGNED_FN)
    for key, value in metrics.items():
        print(f"{key:12s}: {value:.4f}")
