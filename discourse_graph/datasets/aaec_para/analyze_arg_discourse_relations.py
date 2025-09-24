import json
import re
from collections import defaultdict, Counter

def parse_rst_tree(tree_pred):
    """Parse RST tree prediction to extract discourse relations between spans."""
    relations = []
    # Pattern to match RST relations: (start:role=relation:end)
    pattern = re.compile(r'\((\d+):(\w+)=([^:]+):(\d+),(\d+):(\w+)=([^:]+):(\d+)\)')
    
    for match in pattern.finditer(tree_pred):
        span1_start, role1, rel1, span1_end = match.groups()[:4]
        span2_start, role2, rel2, span2_end = match.groups()[4:]
        
        # Extract the actual relation label (skip "span" relations)
        relation = None
        if rel1 != "span":
            relation = rel1
        elif rel2 != "span":
            relation = rel2
            
        if relation:
            relations.append({
                'relation': relation,
                'span1': (int(span1_start), int(span1_end)),
                'span2': (int(span2_start), int(span2_end)),
                'roles': (role1, role2)
            })
    
    return relations

def get_segment_boundaries(segmentation_pred, total_tokens):
    """Convert segmentation predictions to actual token boundaries."""
    boundaries = [0] + segmentation_pred + [total_tokens - 1]
    segments = []
    for i in range(len(boundaries) - 1):
        segments.append((boundaries[i], boundaries[i + 1]))
    return segments

def spans_overlap(span1, span2):
    """Check if two spans overlap."""
    start1, end1 = span1
    start2, end2 = span2
    return not (end1 < start2 or end2 < start1)

def analyze_argumentative_discourse_relations(arg_file, rst_file):
    """Analyze correspondence between argumentative and discourse relations."""
    
    # Load argumentative data
    with open(arg_file, 'r') as f:
        arg_data = json.load(f)
    
    # Load RST data
    with open(rst_file, 'r') as f:
        rst_data = json.load(f)
    
    # Statistics
    stats = {
        'total_arg_relations': 0,
        'total_discourse_relations': 0,
        'arg_relations_with_discourse': 0,
        'discourse_relations_by_type': Counter(),
        'arg_discourse_correspondence': defaultdict(list),
        'coverage_by_discourse_type': Counter(),
        'total_arg_pairs': 0
    }
    
    # Ensure both datasets have same length
    min_len = min(len(arg_data), len(rst_data))
    
    for i in range(min_len):
        arg_item = arg_data[i]
        rst_item = rst_data[i]
        
        # Get argumentative components and relations
        components = arg_item['components']
        arg_relations = arg_item['relations']
        
        # Get discourse relations from RST parsing
        tree_preds = rst_item.get('all_tree_parsing_pred', [])
        segmentation = rst_item.get('all_segmentation_pred', [])
        total_tokens = len(rst_item.get('input_sentences', []))
        
        # Get segment boundaries
        segments = get_segment_boundaries(segmentation, total_tokens)
        
        # Parse all discourse relations
        discourse_relations = []
        for tree_pred in tree_preds:
            discourse_relations.extend(parse_rst_tree(tree_pred))
        
        stats['total_arg_relations'] += len(arg_relations)
        stats['total_discourse_relations'] += len(discourse_relations)
        
        # Count discourse relations by type
        for disc_rel in discourse_relations:
            stats['discourse_relations_by_type'][disc_rel['relation']] += 1
        
        # Analyze correspondence between argumentative and discourse relations
        for arg_rel in arg_relations:
            head_comp = components[arg_rel['head']]
            tail_comp = components[arg_rel['tail']]
            
            stats['total_arg_pairs'] += 1
            
            # Find which segments contain these argumentative components
            head_segments = []
            tail_segments = []
            
            for seg_idx, (seg_start, seg_end) in enumerate(segments):
                # Check if argumentative component overlaps with this segment
                # Note: We need to map character positions to token positions
                # For simplicity, we'll check if segments have any discourse relations
                head_segments.append(seg_idx)
                tail_segments.append(seg_idx)
            
            # Find discourse relations between segments containing arg components
            found_discourse = False
            discourse_types_found = []
            
            for disc_rel in discourse_relations:
                span1_idx, span2_idx = disc_rel['span1'][0] - 1, disc_rel['span2'][0] - 1  # Convert to 0-based
                
                # Check if this discourse relation connects segments with our arg components
                if (span1_idx in head_segments and span2_idx in tail_segments) or \
                   (span1_idx in tail_segments and span2_idx in head_segments):
                    found_discourse = True
                    discourse_types_found.append(disc_rel['relation'])
                    stats['coverage_by_discourse_type'][disc_rel['relation']] += 1
            
            if found_discourse:
                stats['arg_relations_with_discourse'] += 1
                stats['arg_discourse_correspondence'][arg_rel['type']].extend(discourse_types_found)
    
    return stats

def print_statistics(stats, split_name):
    """Print comprehensive statistics."""
    print(f"\n{'='*60}")
    print(f"AAEC-Para {split_name.upper()} Dataset Statistics")
    print(f"{'='*60}")
    
    print(f"\nTotal Statistics:")
    print(f"  Total argumentative relations: {stats['total_arg_relations']}")
    print(f"  Total discourse relations: {stats['total_discourse_relations']}")
    print(f"  Total argumentative pairs: {stats['total_arg_pairs']}")
    print(f"  Arg relations with discourse support: {stats['arg_relations_with_discourse']}")
    
    if stats['total_arg_relations'] > 0:
        coverage_pct = (stats['arg_relations_with_discourse'] / stats['total_arg_relations']) * 100
        print(f"  Coverage percentage: {coverage_pct:.2f}%")
    
    print(f"\nDiscourse Relations by Type:")
    total_disc = sum(stats['discourse_relations_by_type'].values())
    for rel_type, count in stats['discourse_relations_by_type'].most_common():
        pct = (count / total_disc) * 100 if total_disc > 0 else 0
        print(f"  {rel_type}: {count} ({pct:.2f}%)")
    
    print(f"\nDiscourse Relations Supporting Argumentative Relations:")
    total_supporting = sum(stats['coverage_by_discourse_type'].values())
    for rel_type, count in stats['coverage_by_discourse_type'].most_common():
        pct = (count / total_supporting) * 100 if total_supporting > 0 else 0
        print(f"  {rel_type}: {count} ({pct:.2f}%)")
    
    print(f"\nArgumentative-Discourse Correspondence:")
    for arg_type, disc_types in stats['arg_discourse_correspondence'].items():
        disc_counter = Counter(disc_types)
        print(f"  {arg_type} relations:")
        for disc_type, count in disc_counter.most_common():
            print(f"    -> {disc_type}: {count}")

def main():
    """Main function to analyze all splits."""
    splits = ['train', 'dev', 'test']
    all_stats = {}
    
    for split in splits:
        arg_file = f'aaec_para_{split}.json'
        rst_file = f'aaec_RST_logits_{split}.json'
        
        print(f"Analyzing {split} split...")
        stats = analyze_argumentative_discourse_relations(arg_file, rst_file)
        all_stats[split] = stats
        print_statistics(stats, split)
    
    # Combined statistics
    print(f"\n{'='*60}")
    print(f"COMBINED DATASET STATISTICS")
    print(f"{'='*60}")
    
    combined_stats = {
        'total_arg_relations': sum(s['total_arg_relations'] for s in all_stats.values()),
        'total_discourse_relations': sum(s['total_discourse_relations'] for s in all_stats.values()),
        'arg_relations_with_discourse': sum(s['arg_relations_with_discourse'] for s in all_stats.values()),
        'total_arg_pairs': sum(s['total_arg_pairs'] for s in all_stats.values()),
        'discourse_relations_by_type': Counter(),
        'coverage_by_discourse_type': Counter(),
        'arg_discourse_correspondence': defaultdict(list)
    }
    
    for stats in all_stats.values():
        combined_stats['discourse_relations_by_type'].update(stats['discourse_relations_by_type'])
        combined_stats['coverage_by_discourse_type'].update(stats['coverage_by_discourse_type'])
        for arg_type, disc_types in stats['arg_discourse_correspondence'].items():
            combined_stats['arg_discourse_correspondence'][arg_type].extend(disc_types)
    
    print_statistics(combined_stats, 'combined')

if __name__ == "__main__":
    main()
