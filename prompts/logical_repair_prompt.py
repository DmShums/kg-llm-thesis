"""
run the logical repair first (so you get concrete repair plans / conflict explanations), then use an LLM to choose between alternative repair plans based on semantic priors, provenance, and other non-logical signals.
"""

# run logmap repair
# java -jar logmap-matcher.jar --repair existing_mappings.rdf SOURCE.owl TARGET.owl


from utils.onto_object import OntologyEntryAttr
from prompts.prompt_utils import format_hierarchy

import re
from typing import List, Dict


def parse_logmap_repair_output(logmap_output_path: str) -> List[Dict]:
    results: List[Dict] = []

    current_entity = None
    current_plan = None

    with open(logmap_output_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()

            if not line:
                continue

            # ENTITY line
            if line.startswith("http") and "," not in line:
                if current_entity:
                    results.append(current_entity)

                current_entity = {
                    "entity": line,
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
                    current_plan["mappings"].append({
                        "source": parts[0].strip(),
                        "target": parts[1].strip(),
                        "decision": int(parts[2].strip())
                    })

    if current_entity:
        results.append(current_entity)

    return results


def repair_plan_prompt(conflict_entity: str, repair_plans: List[Dict]) -> str:
    prefix_prompt = f"""
You are an expert in ontology alignment.

We are analysing alternative logical repair plans generated to resolve a coherence conflict
involving the following ontology entity:

FOCAL ENTITY:
{conflict_entity}

Each repair plan proposes removing one or more mappings in order to restore logical coherence.

Important:
All plans are logically valid (they restore coherence).
Your task is to rank them based on semantic and structural considerations.

Ranking criteria (in order of importance):

1) Prefer plans that remove fewer mappings.
2) Prefer plans that preserve mappings with higher confidence.
3) Prefer plans that keep semantically plausible mappings (strong label/definition match).
4) Prefer plans that minimally disrupt the alignment structure.

Return ONLY a JSON array sorted from best to worst:

[
  {{
    "plan_id": <int>,
    "score": <float between 0 and 1>,
    "reason": "<short explanation>"
  }}
]
"""
    
    plan_descriptions = []

    for plan in repair_plans:
        mappings_text = "\n".join(
            [
                f"      - REMOVE: {m['source']}  <->  {m['target']}  (decision={m['decision']})"
                for m in plan["mappings"]
            ]
        )

        plan_block = f"""
Plan {plan['plan_id']}:
  - removed_mappings: {plan['size']}
  - conflict_score: {plan['conflict_score']}
  - confidence_of_removed_mappings: {plan['confidence']}
  - mappings_to_remove:
{mappings_text}
"""
        plan_descriptions.append(plan_block)

    return prefix_prompt + "\n".join(plan_descriptions)




if __name__ == "__main__":
    plans = parse_logmap_repair_output(
        "data/LogMap_repair_plans/anatomy_repairs.txt"
    )

    first_block = plans[0]

    prompt = repair_plan_prompt(
        conflict_entity=first_block["entity"],
        repair_plans=first_block["plans"]
    )

    print(prompt)