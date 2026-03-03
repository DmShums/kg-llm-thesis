"""
run the logical repair first (so you get concrete repair plans / conflict explanations), then use an LLM to choose between alternative repair plans based on semantic priors, provenance, and other non-logical signals.
"""


from utils.onto_object import OntologyEntryAttr, OntologyAccess
from utils.prompts.prompts import prompt_all_data_single_entity

from typing import List, Dict, Optional
import re


def parse_logmap_repair_output(
    logmap_output_path: str,
    o1_path: str,
    o2_path: str
) -> List[Dict]:

    onto_src = OntologyAccess(o1_path, annotate_on_init=True)
    onto_tgt = OntologyAccess(o2_path, annotate_on_init=True)

    results: List[Dict] = []

    current_entity = None
    current_plan = None

    def resolve_entity(uri: str) -> Optional[OntologyEntryAttr]:
        """
        Try to resolve the URI in target ontology first,
        then in source ontology.
        """
        try:
            return OntologyEntryAttr(uri, onto=onto_tgt)
        except Exception:
            try:
                return OntologyEntryAttr(uri, onto=onto_src)
            except Exception:
                return None

    with open(logmap_output_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()

            if not line:
                continue

            # ENTITY line
            if line.startswith("http") and "," not in line:
                if current_entity:
                    results.append(current_entity)

                entity_uri = line
                entity_obj = resolve_entity(entity_uri)

                current_entity = {
                    "entity_uri": entity_uri,
                    "entity_obj": entity_obj,
                    "plans": []
                }

                current_plan = None
                continue

            # PLAN line
            if line.startswith("Plan"):
                match = re.search(
                    r"Plan\s+(\d+)\s+of\s+size:\s+(\d+)\s+conflict\s+score:\s+(\d+)\s+confidence:\s+([0-9.eE+-]+)",
                    line
                )

                if match:
                    current_plan = {
                        "plan_id": int(match.group(1)),
                        "size": int(match.group(2)),
                        "conflict_score": int(match.group(3)),
                        "confidence": float(match.group(4)),
                        "mappings": []
                    }

                    current_entity["plans"].append(current_plan)

                continue

            # MAPPING line
            if "," in line and current_plan:
                parts = line.split(",")

                if len(parts) == 3:
                    source_uri = parts[0].strip()
                    target_uri = parts[1].strip()
                    decision = int(parts[2].strip())

                    source_obj = resolve_entity(source_uri)
                    target_obj = resolve_entity(target_uri)

                    current_plan["mappings"].append({
                        "source_uri": source_uri,
                        "target_uri": target_uri,
                        "source_obj": source_obj,
                        "target_obj": target_obj,
                        "decision": decision
                    })

    if current_entity:
        results.append(current_entity)

    return results


def repair_plan_prompt(
    conflict_entity: OntologyEntryAttr,
    repair_plans: list
) -> str:

    entity_data = prompt_all_data_single_entity(conflict_entity)

    prefix_prompt = f"""
You are an expert in ontology alignment and logical repair.

We are analysing alternative logical repair plans generated to resolve
a coherence conflict involving the following ontology entity.

==============================
FOCAL ENTITY
==============================
CLASS: {conflict_entity.thing_class}

ENTITY DATA:
{entity_data}

==============================
REPAIR PLANS
==============================
"""

    plan_descriptions = []

    for plan in repair_plans:

        mappings_blocks = []

        for idx, m in enumerate(plan["mappings"]):

            source_context = ""
            target_context = ""

            if m.get("source_obj"):
                source_context = prompt_all_data_single_entity(m["source_obj"])

            if m.get("target_obj"):
                target_context = prompt_all_data_single_entity(m["target_obj"])

            mapping_block = f"""
{idx+1}) {m['source_uri']} <-> {m['target_uri']} (decision={m['decision']})

SOURCE ENTITY CONTEXT:
{source_context}

TARGET ENTITY CONTEXT:
{target_context}
"""
            mappings_blocks.append(mapping_block.strip())

        mappings_text = "\n\n".join(mappings_blocks)

        plan_block = f"""
----------------------------------------
Plan {plan['plan_id']}
----------------------------------------
Removed mappings count: {plan['size']}
Conflict score: {plan['conflict_score']}
Confidence of removed mappings: {plan['confidence']}

Mappings to remove:
{mappings_text}
"""

        plan_descriptions.append(plan_block.strip())

    instructions = """
==============================
INSTRUCTIONS
==============================

Rank the repair plans from BEST to WORST.

When ranking:
- Prefer plans that remove fewer mappings.
- Prefer plans that preserve high-confidence mappings.
- Prefer plans that keep semantically plausible mappings
  (based on labels, synonyms, and hierarchy).
- Prefer plans that minimally disrupt the overall alignment structure.

Return ONLY a JSON array in the following format:

[
  {
    "plan_id": <int>,
    "score": <float between 0 and 1>,
    "reason": "<short explanation>"
  }
]

Do not include any extra text outside the JSON.
"""

    return prefix_prompt + "\n\n".join(plan_descriptions) + instructions