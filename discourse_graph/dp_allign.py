import re
from evaluate import BatchEvaluator
from utils import decode_anl

def tokenize(text):
    return text.strip().split()

def extract_spans(text):
    return [(m.group(0), m.start(), m.end()) for m in
            re.finditer(r"\[\s*([^\[\]]+?)\s*\]", text)]

def parse_span(span_text):
    parts = [p.strip() for p in span_text.strip("[]").split("|")]
    texts = parts[::2]           # e.g. ["…first clause…", "…second clause…"]
    label = parts[1] if len(parts) > 1 else ""
    return texts, label

def word_distance(a, b):
    # maybe switch to Levenshtein or at least lower the threshold
    return sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b))

def align_pred_to_true(true, pred, max_diff):
    true_spans = extract_spans(true)
    pred_spans = extract_spans(pred)

    replacements = []
    for t_span, *_ in true_spans:
        t_texts, t_label = parse_span(t_span)
        t_tokens = [tokenize(t) for t in t_texts]

        for p_span, *_ in pred_spans:
            p_texts, p_label = parse_span(p_span)

            # –– Drop the label‐matching filter if you want to align anyway:
            # if t_label.lower() != p_label.lower():
            #     continue

            # Only compare number of inner segments; if they differ, skip
            if len(t_texts) != len(p_texts):
                continue

            # Compute distance on each piece
            distances = [
                word_distance(tokenize(tt), tokenize(pt))
                for tt, pt in zip(t_texts, p_texts)
            ]
            max_piece_dist = max(distances)

            # Loosen threshold or use Levenshtein
            if max_piece_dist <= max_diff:
                if p_span != t_span:
                    replacements.append((p_span, t_span))
                break

    # Sort by longest old span first
    replacements.sort(key=lambda x: -len(x[0]))

    new_pred = pred
    for old, new in replacements:
        new_pred = new_pred.replace(old, new, 1)

    return new_pred

def process_file(file_path, max_diff):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    pairs = re.findall(r"True:(.*?)\nPred:(.*?)\n-+", content, re.DOTALL)
    output_lines = []
    for true_text, pred_text in pairs:
        aligned_pred = align_pred_to_true(true_text.strip(),
                                          pred_text.strip(),
                                          max_diff)
        output_lines.append(
            f"True:{true_text.strip()}\nPred:{aligned_pred.strip()}\n{'-'*80}"
        )

    return "\n".join(output_lines)

def evaluate_from_aligned_text(filepath, decode_fn):
    evaluator = BatchEvaluator()
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    pairs = re.findall(r"True:(.*?)\nPred:(.*?)\n-+", content, re.DOTALL)
    for true_text, pred_text in pairs:
        decoded_true = decode_fn(true_text.strip())
        decoded_pred = decode_fn(pred_text.strip())
        evaluator.add_batch(
            [decoded_true[0]], [decoded_true[1]],
            [decoded_pred[0]], [decoded_pred[1]]
        )

    return evaluator.evaluate()

if __name__ == "__main__":
    # Try raising max_diff from 2 up to, say, 10 or even 20
    aligned = process_file("t5_best_rgat.txt", max_diff=10)
    with open("aligned_predictions.txt", "w", encoding="utf-8") as f:
        f.write(aligned)
    print("Done! Aligned predictions saved to 'aligned_predictions.txt'")

    metrics = evaluate_from_aligned_text("aligned_predictions.txt", decode_anl)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
