from utils.onto_object import OntologyEntryAttr, OntologyAccess
from modules.llm_validator import LLMValidator
from modules.evaluator import OntologyAlignmentEvaluator
import pandas as pd
from datetime import datetime
import tqdm


DATASETS_MAP = {
    "anatomy": ["human-mouse"],
}

o1_path = "data/anatomy/human-mouse/human.owl"
o2_path = "data/anatomy/human-mouse/mouse.owl"

gt_path = "/Users/shuma/Desktop/dyplom/data/anatomy/human-mouse/refs_equiv/full.tsv"

onto_src = OntologyAccess(o1_path, annotate_on_init=True)
onto_tgt = OntologyAccess(o2_path, annotate_on_init=True)

validator = LLMValidator()

with open("output/logmap_mappings_to_ask_oracle_user_llm.txt", "r") as f:
    lines = f.readlines()

results = []


for line in tqdm.tqdm(lines[:10]):
    src_uri, tgt_uri = line.strip().split("|")[0], line.strip().split("|")[1]

    src_entity = OntologyEntryAttr(class_uri=src_uri, onto=onto_src)
    tgt_entity = OntologyEntryAttr(class_uri=tgt_uri, onto=onto_tgt)

    res = validator.validate(
        src_entity,
        tgt_entity,
        prompt_type="sequential_hierarchy_with_synonyms",
        system_prompt_type="synonym_aware"
    )

    results.append({
        "Source": src_uri,
        "Target": tgt_uri,
        "LLMDecision": res["is_match"],
        "LLMConfidence": res["confidence"],
        "LogMapScore": 1.0,   # placeholder if LogMap score not available
    })

    print(f"Validated pair: {src_uri} -> {tgt_uri}, Result: {res}\n")

df = pd.DataFrame(results)


dataset_name = "anatomy"
experiment_type = "sequential_hierarchy_with_synonyms"

evaluator = OntologyAlignmentEvaluator(gt_path)
report = evaluator.evaluate(
    df=df,
    dataset_name=dataset_name,
    experiment_type=experiment_type,
    prompts_used="sequential_hierarchy_with_synonyms"
)

print("=== Evaluation Report ===")
print(report)

print("=== Sample of Detailed Results ===")
print(pd.read_csv(f"results/{pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M')}/{dataset_name}/{experiment_type}/detailed_results.csv").head())