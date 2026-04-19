from utils.onto_object import OntologyEntryAttr
from utils.prompts.prompt_utils import (
    format_hierarchy,
    select_best_direct_entity_names,
    select_best_direct_entity_names_with_synonyms,
    select_best_sequential_hierarchy_with_synonyms,
    make_few_shot_prompt_with_label,
    get_gt_pairs,
)


# Ontological prompts

def prompt_direct_entity_ontological(src_entity: OntologyEntryAttr, tgt_entity: OntologyEntryAttr) -> str:
    """Ontological prompt that uses ontology-focused language."""
    src_parent, tgt_parent, src_entity_names, tgt_entity_names = select_best_direct_entity_names(src_entity, tgt_entity)

    prompt_lines = [
        "Analyze the following entities, each originating from a distinct biomedical ontology.",
        "Your task is to assess whether they represent the **same ontological concept**, considering both their semantic meaning and hierarchical position.",
        f'\n1. Source entity: "{src_entity_names}"',
        f"\t- Direct ontological parent: {src_parent}",
        f'\n2. Target entity: "{tgt_entity_names}"',
        f"\t- Direct ontological parent: {tgt_parent}",
        '\nAre these entities **ontologically equivalent** within their respective ontologies? Respond with "True" or "False".',
    ]

    return "\n".join(prompt_lines)


def prompt_sequential_hierarchy_ontological(src_entity: OntologyEntryAttr, tgt_entity: OntologyEntryAttr) -> str:
    """Ontological prompt that uses ontology-focused language, and takes hierarchical relationships into account."""
    src_hierarchy = format_hierarchy(src_entity.get_parents_by_levels(max_level=2))
    tgt_hierarchy = format_hierarchy(tgt_entity.get_parents_by_levels(max_level=2))

    prompt_lines = [
        "Analyze the following entities, each originating from a distinct biomedical ontology.",
        "Each is represented by its **ontological lineage**, capturing its hierarchical placement from the most general to the most specific level.",
        f"\n1. Source entity ontological lineage:\n{src_hierarchy}",
        f"\n2. Target entity ontological lineage:\n{tgt_hierarchy}",
        '\nBased on their **ontological positioning, hierarchical relationships, and semantic alignment**, do these entities represent the **same ontological concept**? Respond with "True" or "False".',
    ]
    return "\n".join(prompt_lines)


# Natural language prompts


def prompt_direct_entity(src_entity: OntologyEntryAttr, tgt_entity: OntologyEntryAttr) -> str:
    """Regular prompt that uses natural language and is more intuitive."""
    src_parent, tgt_parent, src_entity_names, tgt_entity_names = select_best_direct_entity_names(src_entity, tgt_entity)
    prompt_lines = [
        "We have two entities from different biomedical ontologies.",
        (
            f'The first one is "{src_entity_names}"'
            + (f', which belongs to the broader category "{src_parent}"' if src_parent else "")
        ),
        (
            f'The second one is "{tgt_entity_names}"'
            + (f', which belongs to the broader category "{tgt_parent}"' if tgt_parent else "")
        ),
        '\nDo they mean the same thing? Respond with "True" or "False".',
    ]
    return "\n".join(prompt_lines)


def prompt_sequential_hierarchy(src_entity: OntologyEntryAttr, tgt_entity: OntologyEntryAttr) -> str:
    """Regular prompt that uses natural language and is more intuitive."""
    src_hierarchy = format_hierarchy(src_entity.get_parents_by_levels(max_level=2), True)
    tgt_hierarchy = format_hierarchy(tgt_entity.get_parents_by_levels(max_level=2), True)

    prompt_lines = [
        "We have two entities from different biomedical ontologies.",
        (
            f'The first one is "{src_hierarchy[0]}"'
            + (f', which belongs to the broader category "{src_hierarchy[1]}"' if len(src_hierarchy) > 1 else "")
            + (f', under the even broader category "{src_hierarchy[2]}"' if len(src_hierarchy) > 2 else "")
        ),
        (
            f'The second one is "{tgt_hierarchy[0]}"'
            + (f', which belongs to the broader category "{tgt_hierarchy[1]}"' if len(tgt_hierarchy) > 1 else "")
            + (f', under the even broader category "{tgt_hierarchy[2]}"' if len(tgt_hierarchy) > 2 else "")
        ),
        '\nDo they mean the same thing? Respond with "True" or "False".',
    ]

    return "\n".join(prompt_lines)


# Natural language prompts with synonyms


def prompt_direct_entity_with_synonyms(src_entity: OntologyEntryAttr, tgt_entity: OntologyEntryAttr) -> str:
    """Natural language prompt that includes synonyms for a more intuitive comparison."""
    src_parent, tgt_parent, src_entity_names, tgt_entity_names, src_synonyms, tgt_synonyms = (
        select_best_direct_entity_names_with_synonyms(src_entity, tgt_entity)
    )

    src_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in src_synonyms)) if src_synonyms else ""

    tgt_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in tgt_synonyms)) if tgt_synonyms else ""
    prompt_lines = [
        "We have two entities from different biomedical ontologies.",
        f'The first one is "{src_entity_names}"{src_synonyms_text}, which falls under the category "{src_parent}".',
        f'The second one is "{tgt_entity_names}"{tgt_synonyms_text}, which falls under the category "{tgt_parent}".',
        '\nDo they mean the same thing? Respond with "True" or "False".',
    ]
    return "\n".join(prompt_lines)


def prompt_sequential_hierarchy_with_synonyms(src_entity: OntologyEntryAttr, tgt_entity: OntologyEntryAttr) -> str:
    """Generate a natural language prompt asking whether two ontology entities (with synonyms and hierarchy).

    Represent the same concept (True/False).
    """
    src_hierarchy = format_hierarchy(src_entity.get_parents_by_levels(max_level=2), True)
    tgt_hierarchy = format_hierarchy(tgt_entity.get_parents_by_levels(max_level=2), True)

    src_syns, tgt_syns, src_parents_syns, tgt_parents_syns = select_best_sequential_hierarchy_with_synonyms(
        src_entity, tgt_entity, max_level=2
    )

    def describe_entity(hierarchy: list[str], entity_syns: list[str], parent_syns: list[list[str]]) -> str:
        # Base name and its synonyms
        name_part = f'"{hierarchy[0]}"'
        if entity_syns:
            alt = ", ".join(f'"{s}"' for s in entity_syns)
            name_part += f", also known as {alt}"

        parts = [name_part]

        labels = ["belongs to broader category", "under the even broader category", "under the even broader category"]
        for i, parent_name in enumerate(hierarchy[1:]):
            text = f'{labels[i]} "{parent_name}"'
            if parent_syns[i]:
                alt = ", ".join(f'"{s}"' for s in parent_syns[i])
                text += f" (also known as {alt})"
            parts.append(text)

        return ", ".join(parts)

    src_desc = describe_entity(src_hierarchy, src_syns, src_parents_syns)
    tgt_desc = describe_entity(tgt_hierarchy, tgt_syns, tgt_parents_syns)

    prompt_lines = [
        "We have two entities from different biomedical ontologies.",
        f"The first one is {src_desc}.",
        f"The second one is {tgt_desc}.",
        '\nDo they mean the same thing? Respond with "True" or "False".',
    ]
    return "\n".join(prompt_lines)

# New children prompts
def prompt_direct_entity_children(src_entity: OntologyEntryAttr, tgt_entity: OntologyEntryAttr) -> str:
    """Prompt including direct children and direct parent info for LLM comparison."""
    
    # Get direct parent names (take first preferred name if multiple)
    src_parent = next(iter(src_entity.get_direct_parent().get_preffered_names()), None) if src_entity.get_direct_parent() else None
    tgt_parent = next(iter(tgt_entity.get_direct_parent().get_preffered_names()), None) if tgt_entity.get_direct_parent() else None

    # Get direct children names (take first preferred name per child)
    src_children = [next(iter(c.get_preffered_names())) for c in src_entity.get_direct_children()]
    tgt_children = [next(iter(c.get_preffered_names())) for c in tgt_entity.get_direct_children()]
    # print(f"DEBUG: src_parent={src_parent}, src_children={src_children}")
    # print(f"DEBUG: tgt_parent={tgt_parent}, tgt_children={tgt_children}")

    prompt_lines = [
        "We have two entities from different biomedical ontologies.",
        f'The first one is "{next(iter(src_entity.get_preffered_names()))}"' + 
            (f', which belongs to the broader category "{src_parent}"' if src_parent else "") +
            (f' and has children: {", ".join(src_children)}' if src_children else ""),
        f'The second one is "{next(iter(tgt_entity.get_preffered_names()))}"' + 
            (f', which belongs to the broader category "{tgt_parent}"' if tgt_parent else "") +
            (f' and has children: {", ".join(tgt_children)}' if tgt_children else ""),
        '\nDo they mean the same thing? Respond with "True" or "False".',
    ]
    
    return "\n".join(prompt_lines)

def prompt_direct_entity_children_no_parents(src_entity: OntologyEntryAttr, tgt_entity: OntologyEntryAttr) -> str:
    """Prompt including direct children but ignoring parent info."""
    
    # Direct children names
    src_children = [next(iter(c.get_preffered_names())) for c in src_entity.get_direct_children()]
    tgt_children = [next(iter(c.get_preffered_names())) for c in tgt_entity.get_direct_children()]
    # print(f"DEBUG: src_children={src_children}")
    # print(f"DEBUG: tgt_children={tgt_children}")

    prompt_lines = [
        "We have two entities from different biomedical ontologies.",
        f'The first one is "{next(iter(src_entity.get_preffered_names()))}"' +
            (f' and has children: {", ".join(src_children)}' if src_children else ""),
        f'The second one is "{next(iter(tgt_entity.get_preffered_names()))}"' +
            (f' and has children: {", ".join(tgt_children)}' if tgt_children else ""),
        '\nDo they mean the same thing? Respond with "True" or "False".',
    ]
    
    return "\n".join(prompt_lines)

# Subsumption prompts

def prompt_source_subsumed_by_target(src_entity: OntologyEntryAttr, tgt_entity: OntologyEntryAttr) -> str:
    """Prompt asking whether the source entity is a subtype of the target entity (source ⊑ target)."""
    src_parent = next(iter(src_entity.get_direct_parent().get_preffered_names()), None) if src_entity.get_direct_parent() else None
    tgt_parent = next(iter(tgt_entity.get_direct_parent().get_preffered_names()), None) if tgt_entity.get_direct_parent() else None

    prompt_lines = [
        "We have two entities from different biomedical ontologies.",
        f'The first one is "{next(iter(src_entity.get_preffered_names()))}"' +
            (f', which belongs to the broader category "{src_parent}"' if src_parent else ""),
        f'The second one is "{next(iter(tgt_entity.get_preffered_names()))}"' +
            (f', which belongs to the broader category "{tgt_parent}"' if tgt_parent else ""),
        '\nIs the first entity a more specific type of the second entity? In other words, is the first entity a subclass of the second entity? Respond with "True" or "False".',
    ]

    return "\n".join(prompt_lines)


def prompt_target_subsumed_by_source(src_entity: OntologyEntryAttr, tgt_entity: OntologyEntryAttr) -> str:
    """Prompt asking whether the target entity is a subtype of the source entity (target ⊑ source)."""
    # Just swap roles
    return prompt_source_subsumed_by_target(tgt_entity, src_entity)


# -------------------------------
#        Few-Shot Prompts
# -------------------------------

def dummy_few_shot_prompt(
    prompt_func,
    gt_dataset_name: str,
    gt_set_name: str,
    gt_onto_src,
    gt_onto_tgt
) -> str:
    gt_pairs = get_gt_pairs(gt_dataset_name, gt_set_name)

    # Manually selected pairs for few-shot learning
    gt_pairs_for_true = gt_pairs[5:6]
    gt_pairs_for_false = gt_pairs[7:9]
    gt_pairs_for_false = [
        (gt_pairs_for_false[0][0], gt_pairs_for_false[1][1]),
        (gt_pairs_for_false[1][0], gt_pairs_for_false[0][1])
    ][:1]

    # Create labeled prompts
    few_shot_true = make_few_shot_prompt_with_label(
        prompt_func=prompt_func,
        few_shot_pairs=gt_pairs_for_true,
        onto_src=gt_onto_src,
        onto_tgt=gt_onto_tgt,
        label="True",
    )

    few_shot_false = make_few_shot_prompt_with_label(
        prompt_func=prompt_func,
        few_shot_pairs=gt_pairs_for_false,
        onto_src=gt_onto_src,
        onto_tgt=gt_onto_tgt,
        label="False",
    )

    full_prompt = few_shot_true + few_shot_false
    return full_prompt


# -------------------------------
#        Single Entity Prompts
# -------------------------------

def prompt_all_data_single_entity(entity: OntologyEntryAttr) -> str:
    return f"""
- All Entity names: {entity.get_preffered_names()}
- Parent Entity Namings: {entity.get_parents_preferred_names()}
- Entity Synonyms: {entity.get_synonyms()}
""".strip()