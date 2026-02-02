from utils.rag import bioportal_recommender
import os
from pathlib import Path
import json
from utils.onto_access import OntologyAccess
from utils.rag import chat_invoke, bioportal_download_latest_submission

def extract_representative_terms(onto_access, n_top=5, n_mid=10, n_leaf=15):
        classes = list(onto_access.getClasses())

        def depth(cls):
            """
            If a class has many ancestors –> it is deep / specific
            If it has few -> generic / high-level
            """

            # approximate depth: length of shortest path to owl:Thing
            try:
                return len(cls.ancestors())
            except:
                return 0

        buckets = {
            "top": [],
            "mid": [],
            "leaf": []
        }

        for cls in classes:
            d = depth(cls)
            if d <= 2:
                buckets["top"].append(cls)
            elif d <= 5:
                buckets["mid"].append(cls)
            else:
                buckets["leaf"].append(cls)

        import random

        selected = []
        for bucket, k in [("top", n_top), ("mid", n_mid), ("leaf", n_leaf)]:
            if buckets[bucket]:
                selected.extend(random.sample(buckets[bucket], min(k, len(buckets[bucket]))))

        # Extract labels
        terms = []
        for cls in selected:
            label = getattr(cls, "label", None)
            if label:
                if isinstance(label, list):
                    terms.append(str(label[0]))
                else:
                    terms.append(str(label))
            else:
                terms.append(cls.name.replace("_", " "))

        return list(set(terms))

def select_representative_terms_llm(seed_terms, max_terms=10):
    """
    Use LLM to select representative terms from a list of seed terms.
    """

    prompt = f"""
Given the following list of ontology terms:
{', '.join(seed_terms)}
Select the {max_terms} most representative terms that best capture the core concepts of the ontology.
Return the selected terms as a comma-separated list.
"""
    messages = [
        {"role": "system", "content": "You are an expert in ontology analysis."},
        {"role": "user", "content": prompt}
    ]
    response = chat_invoke(
        messages=messages
    )

    return response


if __name__ == "__main__":
    o1_path = "data/anatomy/human-mouse/human.owl"
    o2_path = "data/anatomy/human-mouse/mouse.owl"

    o1 = OntologyAccess(o1_path)
    o2 = OntologyAccess(o2_path)
    
    seed_terms_o1 = extract_representative_terms(o1)
    seed_terms_o2 = extract_representative_terms(o2)

    print("Representative terms from O1:")
    for term in seed_terms_o1:
        print(f"- {term}")

    print("\nLLM-selected representative terms from O1:")
    seed_terms_o1 = select_representative_terms_llm(seed_terms_o1)
    print(seed_terms_o1)

    print("\nRepresentative terms from O2:")
    for term in seed_terms_o2:
        print(f"- {term}")
    seed_terms_o2 = select_representative_terms_llm(seed_terms_o2)
    print("\nLLM-selected representative terms from O2:")
    print(seed_terms_o2)

    out_dir = Path("output/mediating_example")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # Step A: Pick a mediating ontology (MO) with BioPortal Recommender
    # ---------------------------------------------------------------------
    seed_text = "O1 terms:\n" + "\n".join(seed_terms_o1) + "\n\nO2 terms:\n" + "\n".join(seed_terms_o2)


    recs = bioportal_recommender(seed_text)
    print("Top BioPortal Recommender results for mediating ontology:")

    with open(out_dir / "bioportal_recommender_results.json", "w") as f:
        json.dump(recs, f, indent=2)
    
    print(f"Results saved to {out_dir / 'bioportal_recommender_results.json'}")
    print("Retrieved recommendations:")

    for rec in recs:
        score = rec.get("evaluationScore", 0.0)
        ontos = rec.get("ontologies", [])
        if not ontos:
            continue

        onto = ontos[0]
        acronym = onto.get("acronym", "UNKNOWN")
        onto_id = onto.get("@id", "NO_ID")

        print(f"  {acronym} - {onto_id} ({score:.4f})")
    
    
    print("\nDownloading top mediating ontology...")
    top_onto = recs[0]["ontologies"][0]
    top_acronym = top_onto["acronym"]
    out_path = out_dir / f"{top_acronym}.owl"

    bioportal_download_latest_submission(
        top_acronym,
        str(out_path),
        fallback_ontology_json=top_onto  # use the JSON from recommender
    )

    print(f"Mediating ontology saved to {out_path}")