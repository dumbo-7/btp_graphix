import re
import string

# def generate_anl_end_to_end(text, components, relations, entities) -> str:
#     # Add IDs to components
#     index_counter = 0
#     for component in components:
#         component['id'] = index_counter
#         index_counter += 1

#     # Sort components by their start index
#     sorted_components = sorted(components, key=lambda x: x['start'])

#     # Create a dictionary to track which component has which relations
#     relation_dict = {}
#     for relation in relations:
#         relation_type = relation['type']
#         head = relation['head']
#         tail = relation['tail']
#         if head not in relation_dict:
#             relation_dict[head] = []
#         relation_dict[head].append((relation_type, tail))

#     # Generate the formatted output
#     formatted_output = ""
#     prev_end = 0  # Track the end of the previous span

#     for comp in sorted_components:
#         comp_index, comp_type, comp_start, comp_end = comp['id'], comp['type'], comp['start'], comp['end']

#         # Add text before the component span
#         formatted_output += text[prev_end:comp_start]

#         component_text = text[comp_start:comp_end]
#         formatted_output += f"[ {component_text} | {comp_type} "

#         # Add relations if any
#         if comp_index in relation_dict:
#             for relation_type, tail in relation_dict[comp_index]:
#                 tail_component = next(filter(lambda x: x['id'] == tail, components))
#                 tail_text = text[tail_component['start']:tail_component['end']]
#                 formatted_output += f"| {relation_type.capitalize()} = {tail_text} "

#         formatted_output += "]"
#         prev_end = comp_end

#     # Add any remaining text after the last component
#     formatted_output += text[prev_end:]

#     target_output = " ".join(formatted_output.split())  # Remove extra spaces

#     # ADD OUTCOMES ENTITIES
#     outcome_entities = [entity for item in entities if item['type'] == 'outcome' for entity in item['entity']]
#     outcome_string = ', '.join(outcome_entities)
#     outcomes = f"Outcomes: (({outcome_string}))"

#     target_output = f"{target_output}\n{outcomes}"

#     print (target_output)
#     print ("\n")
#     return target_output


def tanl_end_to_end(text, components, relations) -> str:
    # Add IDs to components
    index_counter = 0
    for component in components:
        component['id'] = index_counter
        index_counter += 1

    # Sort components by their start index
    sorted_components = sorted(components, key=lambda x: x['start'])

    # Create a dictionary to track which component has which relations
    relation_dict = {}
    for relation in relations:
        relation_type = relation['type']
        head = relation['head']
        tail = relation['tail']
        if head not in relation_dict:
            relation_dict[head] = []
        relation_dict[head].append((relation_type, tail))

    # Generate the formatted output
    formatted_output = ""
    prev_end = 0  # Track the end of the previous span

    for comp in sorted_components:
        comp_index, comp_type, comp_start, comp_end = comp['id'], comp['type'], comp['start'], comp['end']

        # Add text before the component span
        formatted_output += text[prev_end:comp_start]

        component_text = text[comp_start:comp_end]
        formatted_output += f"[ {component_text} | {comp_type} "

        # Add relations if any
        if comp_index in relation_dict:
            for relation_type, tail in relation_dict[comp_index]:
                tail_component = next(filter(lambda x: x['id'] == tail, components))
                tail_text = text[tail_component['start']:tail_component['end']]
                formatted_output += f"| {relation_type.capitalize()} = {tail_text} "

        formatted_output += "]"
        prev_end = comp_end

    # Add any remaining text after the last component
    formatted_output += text[prev_end:]

    target_output = " ".join(formatted_output.split())  # Remove extra spaces

    return target_output

def anl_input(text, components) -> str:

    index_counter = 0
    for component in components:
        component['id'] = index_counter
        index_counter += 1

    # Sort components by their start index
    sorted_components = sorted(components, key=lambda x: x['start'])

    # Generate the formatted input
    formatted_input = ""
    prev_end = 0  # Track the end of the previous span

    for comp in sorted_components:
        comp_start, comp_end = comp['start'], comp['end']

        # Add text before the component span
        formatted_input += text[prev_end:comp_start]

        component_text = text[comp_start:comp_end]
        formatted_input += f"[ {component_text} "

        formatted_input += "]"
        prev_end = comp_end

    # Add any remaining text after the last component
    formatted_input += text[prev_end:]

    target_input = " ".join(formatted_input.split())  # Remove extra spaces

    # print (target_input)
    return target_input


def generate_anl_end_to_end(text, components, relations) -> str:
    # Add IDs to components
    index_counter = 0
    for component in components:
        component['id'] = index_counter
        index_counter += 1

    # Sort components by their start index
    sorted_components = sorted(components, key=lambda x: x['start'])

    # Create a dictionary to track which component has which relations
    relation_dict = {}
    for relation in relations:
        relation_type = relation['type']
        head = relation['head']
        tail = relation['tail']
        if head not in relation_dict:
            relation_dict[head] = []
        relation_dict[head].append((relation_type, tail))

    # Generate the formatted output
    formatted_output = ""
    prev_end = 0  # Track the end of the previous span

    for comp in sorted_components:
        comp_index, comp_type, comp_start, comp_end = comp['id'], comp['type'], comp['start'], comp['end']

        # Add text before the component span
        formatted_output += text[prev_end:comp_start]

        component_text = text[comp_start:comp_end]
        formatted_output += f"[ {component_text} | {comp_type} "

        # Add relations if any
        if comp_index in relation_dict:
            for relation_type, tail in relation_dict[comp_index]:
                tail_component = next(filter(lambda x: x['id'] == tail, components))
                tail_text = text[tail_component['start']:tail_component['end']]
                formatted_output += f"{relation_type} | {tail_text} "

        formatted_output += "]"
        prev_end = comp_end

    # Add any remaining text after the last component
    formatted_output += text[prev_end:]

    target_output = " ".join(formatted_output.split())  # Remove extra spaces

    # print (f"Target ANL: {target_output}\n")

    return target_output

def generate_anl_end_to_end_v2(text: str,
                               components: list,
                               relations: list) -> str:
    """
    Args
    ----
    text        : full essay string
    components  : [{start, end, type}, ...]
    relations   : [{head, tail, type}, ...]  # head & tail are numeric indices
    Returns
    -------
    string in the format:
        [ span | Type | ID | RelType = ID ... ] …
    """

    # ---- 1. number the components per TYPE --------------
    prefix_map = {"Claim": "C",
                  "MajorClaim": "MC",
                  "Premise": "P"}
    counters = {p: 0 for p in prefix_map.values()}

    for idx, comp in enumerate(components):
        comp["idx"] = idx                       # numeric index (head / tail ref)
        p = prefix_map.get(comp["type"], comp["type"][0].upper())
        counters[p] += 1
        comp["uid"] = f"{p}{counters[p]}"

    # ---- 2. relation lookup -----------------------------
    rel_dict = {}
    for rel in relations:          # {'head': int, 'tail': int, 'type': str}
        rel_dict.setdefault(rel["head"], []).append(rel)

    # ---- 3. compose output ------------------------------
    out, prev_end = [], 0
    for comp in sorted(components, key=lambda c: c["start"]):

        # text before this component
        out.append(text[prev_end:comp["start"]])

        # core fields
        span_txt = text[comp["start"]:comp["end"]]
        buf = [f"[ {span_txt} ", f"| {comp['type']} ", f"| {comp['uid']} "]

        # add any relations that originate from this component
        for r in rel_dict.get(comp["idx"], []):
            tail_uid = components[r["tail"]]["uid"]
            rel_name = r["type"].title()          # “supports” ➜ “Supports”
            buf.append(f"| {rel_name} = {tail_uid} ")

        buf.append("]")
        out.append("".join(buf))

        prev_end = comp["end"]

    out.append(text[prev_end:])                   # trailing text
    return " ".join("".join(out).split())         # squeeze whitespace


def prepare_data(dataset):
    input_sentences = []
    target_sentences = []

    for example in dataset:
        input_sen = example["paragraph"]
        # input_sen = anl_input(example["paragraph"], example["components"])
        target_sen = new_tanl_output(example["paragraph"], example["components"], example["relations"])
        # print (f"True:{target_sen}\nPred:{target_sen}\n{'-'*80}\n")

        input_sentences.append(input_sen)
        target_sentences.append(target_sen)

    return input_sentences, target_sentences


# # Post-processing the anl structure------------------------------------------------------------

# def decode_anl(formatted_text):

#     formatted_text = re.sub(r'\](\W)', r'] \1', formatted_text)

#     comp_pattern = re.compile(r'\[(.*?)\|(.*?)\]')
    
#     # Find all matches of the component pattern in the formatted text
#     matches = comp_pattern.findall(formatted_text)
    
#     components = []
#     relations = []

#     for match in matches:
#         comp_str = match[0].strip()  # Text inside the brackets
#         comp_type_relations = match[1].strip().split('|')  # Type and relations

#         comp_type = comp_type_relations[0].strip()
#         if (len(comp_type.split(" ")) > 1):
#             comp_type = comp_type.split(" ")[0]
            
#         comp_relations = [rel.strip() for rel in comp_type_relations[1:]]

#         # Store the component details
#         components.append({
#             'span': comp_str,
#             'type': comp_type,
#             'relations': comp_relations
#         })

#     # Create relations based on extracted components and relations
#     for component in components:
#         for rel in component['relations']:
#             rel_match = re.match(r'(\w+)\s*=\s*(.*)', rel)
#             if rel_match:
#                 rel_type = rel_match.group(1).strip()
#                 rel_target_span = rel_match.group(2).strip()
                
#                 # Find the target component by span
#                 target_component = next((comp for comp in components if comp['span'] == rel_target_span), None)
#                 if target_component:
#                     relations.append((
#                         (component['span'], component['type']),
#                         rel_type,
#                         (target_component['span'], target_component['type'])
#                     ))

#     component_tuples = [(comp['type'], comp['span']) for comp in components]

#     # print("Extracted components:", component_tuples)
#     # print("Extracted relations:", relations)
#     return component_tuples, relations


def decode_anl(formatted_text):
    formatted_text = re.sub(r'\](\W)', r'] \1', formatted_text)

    comp_pattern = re.compile(r'\[(.*?)\|(.*?)\]')
    
    # Find all matches of the component pattern in the formatted text
    matches = comp_pattern.findall(formatted_text)
    
    components = []
    relations = []

    for match in matches:
        comp_str = match[0].strip()  # Text inside the brackets
        comp_type_relations = match[1].strip().split(' ')  # Type and relations

        comp_type = comp_type_relations[0].strip()
        comp_relations = [" ".join([rel.strip() for rel in comp_type_relations[1:]])]

        # print (comp_relations)

        # Store the component details
        components.append({
            'span': comp_str,
            'type': comp_type,
            'relations': comp_relations
        })

    # Create relations based on extracted components and relations
    for component in components:
        for rel in component['relations']:
            rel_match = re.match(r'(\w+)\s*\|\s*(.*)', rel)
            if rel_match:
                rel_type = rel_match.group(1).strip()
                rel_target_span = rel_match.group(2).strip()
                
                # Find the target component by span
                target_component = next((comp for comp in components if comp['span'] == rel_target_span), None)
                if target_component:
                    relations.append((
                        (component['span'], component['type']),
                        rel_type,
                        (target_component['span'], target_component['type'])
                    ))

    component_tuples = [(comp['type'], comp['span']) for comp in components]

    # print("Extracted components:", component_tuples)
    # print("Extracted relations:", relations)
    return component_tuples, relations

def decode_tanl(formatted_text):

    formatted_text = re.sub(r'\](\W)', r'] \1', formatted_text)

    comp_pattern = re.compile(r'\[(.*?)\|(.*?)\]')
    
    # Find all matches of the component pattern in the formatted text
    matches = comp_pattern.findall(formatted_text)
    
    components = []
    relations = []

    for match in matches:
        comp_str = match[0].strip()  # Text inside the brackets
        comp_type_relations = match[1].strip().split('|')  # Type and relations

        comp_type = comp_type_relations[0].strip()
        if (len(comp_type.split(" ")) > 1):
            comp_type = comp_type.split(" ")[0]
            
        comp_relations = [rel.strip() for rel in comp_type_relations[1:]]

        # Store the component details
        components.append({
            'span': comp_str,
            'type': comp_type,
            'relations': comp_relations
        })

    # Create relations based on extracted components and relations
    for component in components:
        for rel in component['relations']:
            rel_match = re.match(r'(\w+)\s*=\s*(.*)', rel)
            if rel_match:
                rel_type = rel_match.group(1).strip()
                rel_target_span = rel_match.group(2).strip()
                
                # Find the target component by span
                target_component = next((comp for comp in components if comp['span'] == rel_target_span), None)
                if target_component:
                    relations.append((
                        (component['span'], component['type']),
                        rel_type,
                        (target_component['span'], target_component['type'])
                    ))

    component_tuples = [(comp['type'], comp['span']) for comp in components]

    # print("Extracted components:", component_tuples)
    # print("Extracted relations:", relations)
    return component_tuples, relations


def decode_anl_v2(formatted_text):
    """
    Parameters
    ----------
    formatted_text : str
        String in the new bracketed format, e.g.
        "... [ some span | Premise | P2 | Supports = C1 ] ..."

    Returns
    -------
    comp_tuples : list[tuple]
        Each element is (TYPE, SPAN)  — exactly what BatchEvaluator wants.
    rel_tuples  : list[tuple]
        Each element is ((SPAN1, TYPE1), REL_TYPE, (SPAN2, TYPE2)).
    """

    # --- 1. grab every bracketed chunk ------------------------------------
    comp_chunks = re.findall(r'\[(.*?)\]', formatted_text)

    uid2info       = {}      # uid -> (span, type)
    comp_tuples    = []      # final component tuples
    pending_rels   = []      # temp store: (head_uid, rel_type, tail_uid)

    for chunk in comp_chunks:
        parts = [p.strip() for p in chunk.split('|')]

        if len(parts) < 3:
            # malformed bracket — skip
            continue

        span_txt, comp_type, uid = parts[:3]

        # store component
        uid2info[uid] = (span_txt, comp_type)
        comp_tuples.append((comp_type, span_txt))

        # any relations attached to this component?
        for extra in parts[3:]:
            if '=' not in extra:
                continue
            rel_type, tail_uid = [s.strip() for s in extra.split('=', 1)]
            pending_rels.append((uid, rel_type, tail_uid))

    # --- 2. resolve relations to tuple form --------------------------------
    rel_tuples = []
    for head_uid, rel_type, tail_uid in pending_rels:
        if head_uid not in uid2info or tail_uid not in uid2info:
            continue  # skip if either side missing

        head_span, head_type = uid2info[head_uid]
        tail_span, tail_type = uid2info[tail_uid]

        rel_tuples.append(
            ((head_span, head_type), rel_type, (tail_span, tail_type))
        )

    return comp_tuples, rel_tuples

# def new_anl_output(text, components, relations) -> str:
#     # Add IDs to components
#     index_counter = 0
#     for component in components:
#         component['id'] = index_counter
#         index_counter += 1

#     # Sort components by their start index
#     sorted_components = sorted(components, key=lambda x: x['start'])

#     # Create a dictionary to track which component has which relations
#     relation_dict = {}
#     for relation in relations:
#         relation_type = relation['type']
#         head = relation['head']
#         tail = relation['tail']
#         if head not in relation_dict:
#             relation_dict[head] = []
#         relation_dict[head].append((relation_type, tail))

#     # Generate the formatted output
#     formatted_output = ""
#     prev_end = 0  # Track the end of the previous span

#     for comp in sorted_components:
#         comp_index, comp_type, comp_start, comp_end = comp['id'], comp['type'], comp['start'], comp['end']

#         # Add text before the component span
#         before_text = text[prev_end:comp_start]
#         # Add space separation for ALL punctuation in the text before component
#         # Escape punctuation for regex and add spaces around them
#         punct_pattern = f"([{re.escape(string.punctuation)}])"
#         before_text = re.sub(punct_pattern, r' \1 ', before_text)
#         formatted_output += before_text

#         component_text = text[comp_start:comp_end]
        
#         # Add space separation for ALL punctuation inside component text as well
#         punct_pattern = f"([{re.escape(string.punctuation)}])"
#         component_text = re.sub(punct_pattern, r' \1 ', component_text)
#         # Clean up extra spaces within the component
#         component_text = re.sub(r'\s+', ' ', component_text).strip()
        
#         formatted_output += f"[ {component_text} | {comp_type} "

#         # Add relations if any
#         if comp_index in relation_dict:
#             for relation_type, tail in relation_dict[comp_index]:
#                 tail_component = next(filter(lambda x: x['id'] == tail, components))
#                 tail_text = text[tail_component['start']:tail_component['end']]
#                 # Add space separation for ALL punctuation in tail text as well
#                 punct_pattern = f"([{re.escape(string.punctuation)}])"
#                 tail_text = re.sub(punct_pattern, r' \1 ', tail_text)
#                 tail_text = re.sub(r'\s+', ' ', tail_text).strip()
#                 formatted_output += f"{relation_type} | {tail_text} "

#         formatted_output += "]"
        
#         prev_end = comp_end

#     # Add any remaining text after the last component
#     remaining_text = text[prev_end:]
#     # Add space separation for ALL punctuation in remaining text
#     punct_pattern = f"([{re.escape(string.punctuation)}])"
#     remaining_text = re.sub(punct_pattern, r' \1 ', remaining_text)
#     formatted_output += remaining_text

#     # Clean up multiple spaces and ensure proper spacing
#     target_output = re.sub(r'\s+', ' ', formatted_output).strip()

#     print (f"Target ANL: {target_output}\n")

#     return target_output

def new_tanl_output(text, components, relations) -> str:
    # Add IDs to components
    index_counter = 0
    for component in components:
        component['id'] = index_counter
        index_counter += 1

    # Sort components by their start index
    sorted_components = sorted(components, key=lambda x: x['start'])

    # Create a dictionary to track which component has which relations
    relation_dict = {}
    for relation in relations:
        relation_type = relation['type']
        head = relation['head']
        tail = relation['tail']
        if head not in relation_dict:
            relation_dict[head] = []
        relation_dict[head].append((relation_type, tail))

    # Generate the formatted output
    formatted_output = ""
    prev_end = 0  # Track the end of the previous span

    for comp in sorted_components:
        comp_index, comp_type, comp_start, comp_end = comp['id'], comp['type'], comp['start'], comp['end']

        # Add text before the component span
        before_text = text[prev_end:comp_start]
        # Add space separation for ALL punctuation in the text before component
        # Escape punctuation for regex and add spaces around them
        punct_pattern = f"([{re.escape(string.punctuation)}])"
        before_text = re.sub(punct_pattern, r' \1 ', before_text)
        formatted_output += before_text

        component_text = text[comp_start:comp_end]
        
        # Add space separation for ALL punctuation inside component text as well
        punct_pattern = f"([{re.escape(string.punctuation)}])"
        component_text = re.sub(punct_pattern, r' \1 ', component_text)
        # Clean up extra spaces within the component
        component_text = re.sub(r'\s+', ' ', component_text).strip()
        
        formatted_output += f"[ {component_text} | {comp_type} "

        # Add relations if any
        if comp_index in relation_dict:
            for relation_type, tail in relation_dict[comp_index]:
                tail_component = next(filter(lambda x: x['id'] == tail, components))
                tail_text = text[tail_component['start']:tail_component['end']]
                # Add space separation for ALL punctuation in tail text as well
                punct_pattern = f"([{re.escape(string.punctuation)}])"
                tail_text = re.sub(punct_pattern, r' \1 ', tail_text)
                tail_text = re.sub(r'\s+', ' ', tail_text).strip()
                formatted_output += f"| {relation_type.capitalize()} = {tail_text} "

        formatted_output += "]"
        
        prev_end = comp_end

    # Add any remaining text after the last component
    remaining_text = text[prev_end:]
    # Add space separation for ALL punctuation in remaining text
    punct_pattern = f"([{re.escape(string.punctuation)}])"
    remaining_text = re.sub(punct_pattern, r' \1 ', remaining_text)
    formatted_output += remaining_text

    # Clean up multiple spaces and ensure proper spacing
    target_output = re.sub(r'\s+', ' ', formatted_output).strip()
    print (f"Target TANL: {target_output}\n")

    return target_output