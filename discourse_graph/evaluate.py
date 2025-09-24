class BatchEvaluator:
    def __init__(self):
        self.all_true_comp_tuples = []
        self.all_pred_comp_tuples = []
        self.all_true_rel_tuples = []
        self.all_pred_rel_tuples = []

    def add_batch(self, true_comp, true_rel, pred_comp, pred_rel):
        for i in range(len(true_comp)):
            true_comp_tuples = true_comp[i]
            pred_comp_tuples = pred_comp[i]

            self.all_true_comp_tuples.extend(true_comp_tuples)
            self.all_pred_comp_tuples.extend(pred_comp_tuples)

            true_rel_tuples = true_rel[i]
            pred_rel_tuples = pred_rel[i]

            self.all_true_rel_tuples.extend(true_rel_tuples)
            self.all_pred_rel_tuples.extend(pred_rel_tuples)

    def evaluate(self):
        # Convert to sets to remove duplicates and enable direct comparison
        all_true_comp_tuples = list(set(self.all_true_comp_tuples))
        all_pred_comp_tuples = list(set(self.all_pred_comp_tuples))
        all_true_rel_tuples = list(set(self.all_true_rel_tuples))
        all_pred_rel_tuples = list(set(self.all_pred_rel_tuples))

        # Calculate precision, recall, and F1 for components
        correct_comp_tuples = set(all_true_comp_tuples) & set(all_pred_comp_tuples)
        comp_precision = len(correct_comp_tuples) / len(all_pred_comp_tuples) if all_pred_comp_tuples else 0
        comp_recall = len(correct_comp_tuples) / len(all_true_comp_tuples) if all_true_comp_tuples else 0
        comp_f1 = (2 * comp_precision * comp_recall / (comp_precision + comp_recall)) if (comp_precision + comp_recall) else 0

        # Calculate precision, recall, and F1 for relations
        correct_rel_tuples = set(all_true_rel_tuples) & set(all_pred_rel_tuples)
        rel_precision = len(correct_rel_tuples) / len(all_pred_rel_tuples) if all_pred_rel_tuples else 0
        rel_recall = len(correct_rel_tuples) / len(all_true_rel_tuples) if all_true_rel_tuples else 0
        rel_f1 = (2 * rel_precision * rel_recall / (rel_precision + rel_recall)) if (rel_precision + rel_recall) else 0

        # ACI F1: Evaluate components ignoring types
        true_comp_spans = {span for _ , span in all_true_comp_tuples}
        pred_comp_spans = {span for _ , span in all_pred_comp_tuples}
        correct_comp_spans = set(true_comp_spans) & set(pred_comp_spans)
        aci_precision = len(correct_comp_spans) / len(pred_comp_spans) if pred_comp_spans else 0
        aci_recall = len(correct_comp_spans) / len(true_comp_spans) if true_comp_spans else 0
        aci_f1 = (2 * aci_precision * aci_recall / (aci_precision + aci_recall)) if (aci_precision + aci_recall) else 0

        # ARI F1: Evaluate relations ignoring types
        true_rel_spans = {(e1, e2) for (e1, _), _, (e2, _) in all_true_rel_tuples}
        pred_rel_spans = {(e1, e2) for (e1, _), _, (e2, _) in all_pred_rel_tuples}
        correct_rel_spans = set(true_rel_spans) & set(pred_rel_spans)
        ari_precision = len(correct_rel_spans) / len(pred_rel_spans) if pred_rel_spans else 0
        ari_recall = len(correct_rel_spans) / len(true_rel_spans) if true_rel_spans else 0
        ari_f1 = (2 * ari_precision * ari_recall / (ari_precision + ari_recall)) if (ari_precision + ari_recall) else 0

        return {
            'ACI_precision': aci_precision,
            'ACI_recall': aci_recall,
            'ACI_f1': aci_f1,
            'ACC_precision': comp_precision,
            'ACC_recall': comp_recall,
            'ACC_f1': comp_f1,
            'ARI_precision': ari_precision,
            'ARI_recall': ari_recall,
            'ARI_f1': ari_f1,
            'ARC_precision': rel_precision,
            'ARC_recall': rel_recall,
            'ARC_f1': rel_f1,
        }
