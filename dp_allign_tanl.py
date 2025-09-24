import re
from discourse_graph.evaluate import BatchEvaluator
from discourse_graph.utils import decode_tanl
import sys

def tokenize(text):
    """Simple whitespace tokenizer."""
    return text.strip().split()

def extract_spans(text):
    """
    Find all substrings of the form "[ … ]" and return (span_text, start, end).
    span_text still includes the square brackets.
    """
    return [
        (m.group(0), m.start(), m.end())
        for m in re.finditer(r"\[\s*([^\[\]]+?)\s*\]", text)
    ]

def parse_span_new(span_text):
    """
    Input:  span_text = "[ A | Premise | Supports = C ]"
    Output: ( [A, C], role )
       - texts = [ "A", "C" ]   (list of two strings)
       - role  = "Premise"      (string)
    If the format is unexpected (fewer than 3 parts), return (None, None).
    """
    inner = span_text.strip()[1:-1]   # remove "[" and "]"
    parts = [p.strip() for p in inner.split("|")]
    # Expect exactly 3 parts: [primary_clause, role, "Supports = claim_text"]
    if len(parts) < 3:
        return None, None

    primary_clause = parts[0]
    role = parts[1]
    right = parts[2]  # e.g. "Supports = C"
    if "=" not in right:
        return None, None

    claim_text = right.split("=", 1)[1].strip()
    texts = [primary_clause, claim_text]
    return texts, role

def word_distance(a_tokens, b_tokens):
    """
    Compute a simple bag‐of‐words Hamming + length difference:
      sum(mismatches at each position) + abs(len(a) - len(b))
    """
    base = sum(x != y for x, y in zip(a_tokens, b_tokens))
    length_diff = abs(len(a_tokens) - len(b_tokens))
    return base + length_diff

def align_pred_to_true_new(true, pred, max_diff):
    """
    1) extract all spans from 'true' and from 'pred'
    2) for each true_span:
         parse it into two text segments; skip if parsing fails
         loop over every pred_span:
            parse it; skip if parsing fails
            if len of segments differ, continue
            compute word_distance on each segment
            if max distance ≤ max_diff, record replacement
    3) sort replacements by length of pred_span_text descending
    4) replace each old_span with new_span once in pred
    """
    true_spans = extract_spans(true)
    pred_spans = extract_spans(pred)

    replacements = []

    for t_span, *_ in true_spans:
        parsed_t = parse_span_new(t_span)
        if parsed_t == (None, None):
            continue
        t_texts, _ = parsed_t
        t_token_lists = [tokenize(t) for t in t_texts]

        for p_span, *_ in pred_spans:
            parsed_p = parse_span_new(p_span)
            if parsed_p == (None, None):
                continue
            p_texts, _ = parsed_p

            if len(t_texts) != len(p_texts):
                continue

            distances = []
            for tt_tokens, pt in zip(t_token_lists, p_texts):
                distances.append(word_distance(tt_tokens, tokenize(pt)))
            max_piece_dist = max(distances)

            if max_piece_dist <= max_diff and p_span != t_span:
                replacements.append((p_span, t_span))
                break

    replacements.sort(key=lambda x: -len(x[0]))

    new_pred = pred
    for old_span, new_span in replacements:
        new_pred = new_pred.replace(old_span, new_span, 1)

    return new_pred

def process_file_new(file_path, max_diff):
    """
    Reads a file containing repeated blocks of:
      True: ...\nPred: ...\n-----…-----
    and returns a single big string “True:…\nPred: aligned…\n-----…-----” for each pair.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    pairs = re.findall(r"True:(.*?)\nPred:(.*?)\n-+", content, re.DOTALL)
    out_lines = []
    for true_text, pred_text in pairs:
        aligned = align_pred_to_true_new(true_text.strip(),
                                         pred_text.strip(),
                                         max_diff)
        out_lines.append(
            f"True:{true_text.strip()}\n"
            f"Pred:{aligned.strip()}\n"
            + "-" * 80
        )
    return "\n".join(out_lines)

def evaluate_from_aligned_text_new(filepath, decode_fn):
    """
    Reads the aligned file, applies decode_fn (decode_tanl), and evaluates.
    """
    evaluator = BatchEvaluator()
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    pairs = re.findall(r"True:(.*?)\nPred:(.*?)\n-+", content, re.DOTALL)
    for true_text, pred_text in pairs:
        dec_true = decode_fn(true_text.strip())
        dec_pred = decode_fn(pred_text.strip())
        evaluator.add_batch(
            [dec_true[0]],
            [dec_true[1]],
            [dec_pred[0]],
            [dec_pred[1]]
        )
    return evaluator.evaluate()

# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Usage:
      python dp_allign_tanl.py <input_file> <aligned_output_file> <max_diff>
    Example:
      python dp_allign_tanl.py t5_best_rgat_newformat.txt aligned_predictions.txt 5
    """
    if len(sys.argv) != 4:
        print("Usage: python dp_allign_tanl.py <input_file> <aligned_output_file> <max_diff>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    try:
        max_diff = int(sys.argv[3])
    except ValueError:
        print("Error: <max_diff> must be an integer.")
        sys.exit(1)

    aligned_text = process_file_new(input_path, max_diff)
    with open(output_path, "w", encoding="utf-8") as fout:
        fout.write(aligned_text)
    print(f"Aligned predictions saved to '{output_path}'.")

    metrics = evaluate_from_aligned_text_new(output_path, decode_tanl)
    print("Evaluation metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
