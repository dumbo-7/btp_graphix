import re
import json

# List of your JSON file paths
file_paths = [
    'aaec_RST_logits_train.json',
    'aaec_RST_logits_dev.json',
    'aaec_RST_logits_test.json'
]

rst_labels = set()
# Pattern to match RST relation labels
pattern = re.compile(r'(Satellite|Nucleus)=([\w-]+)')


for path in file_paths:
    with open(path, 'r') as f:
        data = json.load(f)
        for entry in data:
            tree_preds = entry.get("all_tree_parsing_pred", [])
            for pred in tree_preds:
                matches = pattern.findall(pred)
                for role, label in matches:
                    if label != "span":  # Exclude structural spans
                        rst_labels.add(label)

# Print all unique RST relation labels found
print("Unique RST relation labels across train/dev/test:")
print(sorted(rst_labels))
