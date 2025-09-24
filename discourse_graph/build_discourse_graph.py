import json
import pickle
import torch
from pathlib import Path
from transformers import T5TokenizerFast
# from graph_processor import RSTGraphProcessor
from reduced_graph_processor import RSTGraphProcessor

ROOT = Path("./datasets/aaec_para")

def load_json(name: str):
    with open(ROOT / name, "rb") as f:
        return json.load(f)

# concatenate train / dev / test RST blobs into a single list, same order as used
rst_train = load_json("aaec_RST_logits_train.json")
rst_dev   = load_json("aaec_RST_logits_dev.json")
rst_test  = load_json("aaec_RST_logits_test.json")
rst_list  = rst_train + rst_dev + rst_test

src_train = load_json("aaec_para_train.json")
src_dev   = load_json("aaec_para_dev.json")
src_test  = load_json("aaec_para_test.json")
src_all   = src_train + src_dev + src_test

assert len(rst_list) == len(src_all), "RST vs source length mismatch"

# SentencePiece tokenizer (T5) – vocab matches ▁ tokens
sp_tokenizer = T5TokenizerFast.from_pretrained("t5-base", use_fast=True)

proc = RSTGraphProcessor(rst_list)
graph_pedia = {}

for sid, example in enumerate(src_all):
    tokens = rst_list[sid]["input_sentences"]  # ▁-prefixed sub-words
    ids = sp_tokenizer.convert_tokens_to_ids(tokens)
    if len(tokens) != len(ids):
        raise ValueError(f"sid={sid} token/ID mismatch: {len(tokens)} vs {len(ids)}")

    gdict = proc.build_graph(sid, tokens)
    if gdict["graph"].num_nodes() != len(ids):
        raise ValueError(f"sid={sid} node/feature mismatch: {gdict['graph'].num_nodes()} vs {len(ids)}")

    gdict["features"] = torch.tensor(ids, dtype=torch.long)
    graph_pedia[sid] = gdict
    print(f"✔ built graph {sid} | nodes={len(ids)}")

with open("graph_pedia_discourse.bin", "wb") as fh:
    pickle.dump(graph_pedia, fh)
print("▶ graph_pedia_discourse.bin saved (", len(graph_pedia), "samples )")
