from transformers import XLMRobertaTokenizerFast, AutoTokenizer
import json, os

# 1. Load your T5 tokenizer
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", use_fast=True)  # or your checkpoint

def extract_edus(tokens, seg_indices, tokenizer):
    """
    Given a list of token-piece strings (`tokens`) and a list of boundary indices,
    return a list of (start_idx, detokenized_EDU_string) tuples.
    """
    boundaries = sorted(set(seg_indices))
    edus, start = [], 0
    for end in boundaries:
        # detokenize the sublist [start:end] into a string
        edu_str = tokenizer.convert_tokens_to_string(tokens[start:end])
        edus.append((start, edu_str))
        start = end
    # final EDU
    if start < len(tokens):
        edu_str = tokenizer.convert_tokens_to_string(tokens[start:])
        edus.append((start, edu_str))
    return edus

def print_edus(json_path, tokenizer):
    data = json.load(open(json_path, encoding="utf-8"))
    instances = data if isinstance(data, list) else list(data.values())

    for i, inst in enumerate(instances):
        toks = inst.get("input_sentences", [])
        seg  = inst.get("all_segmentation_pred", [])
        if not toks or not seg:
            continue
        edus = extract_edus(toks, seg, tokenizer)
        print(f"\nInstance {i} ({len(edus)} EDUs)")
        for s, txt in edus:
            print(f"  [{s}] {txt}")

# finally, call it:
print_edus(
    "./discourse_graph/datasets/aaec_para/aaec_RST_logits_test.json",
    tokenizer
)
