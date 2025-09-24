import pickle
import torch
from transformers import AutoTokenizer

# Configuration
GRAPH_FILE = "/workspace/amit/Rohit/NMD/seq2seq/discourse_graph/graph_pedia_discourse.bin"
GRAPH_ID = 42        # Change this to inspect other graphs
NODE_ID = 10         # Change this to inspect a different node

def main():
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base", use_fast=True)
    # tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base", use_fast=True)

    # Load graphs from file
    with open(GRAPH_FILE, "rb") as f:
        graph_pedia = pickle.load(f)

    # Access specific graph
    if GRAPH_ID not in graph_pedia:
        raise ValueError(f"Graph ID {GRAPH_ID} not found in graph_pedia.")

    gdict = graph_pedia[GRAPH_ID]
    graph = gdict["graph"]
    features = gdict["features"]

    print(f"✅ Loaded graph #{GRAPH_ID}")
    print("  → Num nodes:", graph.num_nodes())
    print("  → Num edges:", graph.num_edges())

    # Check node ID bounds
    if NODE_ID >= graph.num_nodes():
        raise ValueError(f"Node ID {NODE_ID} exceeds node count {graph.num_nodes()}")

    # Read token ID and convert to string
    token_id = features[NODE_ID].item()
    token_str = tokenizer.convert_ids_to_tokens([token_id])[0]

    print(f"\n🔍 Node {NODE_ID} details:")
    print("  → Token ID:", token_id)
    print("  → Token string:", token_str)

    # Print edge info
    print("\n📌 Edges:")
    print("  → Incoming:", graph.in_edges(NODE_ID))
    print("  → Outgoing:", graph.out_edges(NODE_ID))

if __name__ == "__main__":
    main()
