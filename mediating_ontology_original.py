"""
Paper-faithful refactor of mediating ontology selection
Based on: Chen et al. (LogMap + BioPortal, Algorithm 2)

Key properties:
- No LLM usage
- Exact lexical overlap between O1 and O2
- BioPortal SEARCH calls per shared label
- Ranking by synonym coverage (positive hits + avg synonyms)
"""

from pathlib import Path
import json
from collections import defaultdict
from utils.onto_access import OntologyAccess
from utils.rag import bioportal_search, bioportal_download_latest_submission

# ------------------------------------------------------------
# Step 1: Extract exact lexical label overlaps between O1 and O2
# ------------------------------------------------------------

def extract_exact_label_matches(o1, o2):
    def collect_labels(onto):
        labels = set()
        for cls in onto.getClasses():
            label = getattr(cls, "label", None)
            if label:
                if isinstance(label, list):
                    labels.add(str(label[0]).lower())
                else:
                    labels.add(str(label).lower())
        return labels

    labels_o1 = collect_labels(o1)
    labels_o2 = collect_labels(o2)

    shared = labels_o1 & labels_o2
    return list(shared)

# ------------------------------------------------------------
# Utility: extract ontology acronym from BioPortal SEARCH result
# ------------------------------------------------------------

def extract_acronym(result):
    onto_url = result.get("links", {}).get("ontology")
    if not onto_url:
        return None
    return onto_url.rstrip("/").split("/")[-1]

# ------------------------------------------------------------
# Step 2: Query BioPortal SEARCH for mediating candidates
# ------------------------------------------------------------

def collect_mediating_candidates(shared_labels, max_queries=25):
    stats = defaultdict(lambda: {
        "positive_hits": 0,
        "synonym_count": 0,
        "ontology_json": None
    })

    for i, label in enumerate(shared_labels):
        if i >= max_queries:
            break

        results = bioportal_search(label)
        seen_ontologies = set()

        for r in results:
            onto = extract_acronym(r)
            synonyms = r.get("synonym", [])

            if onto and synonyms and onto not in seen_ontologies:
                stats[onto]["positive_hits"] += 1
                stats[onto]["synonym_count"] += len(synonyms)
                seen_ontologies.add(onto)

                # ⬇️ capture ontology metadata once
                if stats[onto]["ontology_json"] is None:
                    stats[onto]["ontology_json"] = r.get("ontology")

    return stats

# ------------------------------------------------------------
# Step 3: Rank mediating ontologies (Algorithm 2, Step 9)
# ------------------------------------------------------------

def rank_mediators(stats, top_k=5):
    ranked = sorted(
        stats.items(),
        key=lambda x: (
            x[1]["positive_hits"],
            x[1]["synonym_count"] / max(x[1]["positive_hits"], 1)
        ),
        reverse=True
    )
    return ranked[:top_k]


if __name__ == "__main__":

    o1_path = "data/anatomy/human-mouse/human.owl"
    o2_path = "data/anatomy/human-mouse/mouse.owl"

    out_dir = Path("output/mediating_paper_faithful")
    out_dir.mkdir(parents=True, exist_ok=True)

    o1 = OntologyAccess(o1_path)
    o2 = OntologyAccess(o2_path)

    print("Extracting exact lexical overlaps...")
    shared_labels = extract_exact_label_matches(o1, o2)
    print(f"Found {len(shared_labels)} shared labels")

    print("Querying BioPortal SEARCH...")
    stats = collect_mediating_candidates(shared_labels)

    ranked = rank_mediators(stats)

    print("Top mediating ontology candidates:")
    results = []
    for rank, (acronym, s) in enumerate(ranked, start=1):
        avg_syn = s["synonym_count"] / max(s["positive_hits"], 1)
        print(f"{rank}. {acronym} | hits={s['positive_hits']} | avg_syn={avg_syn:.2f}")
        results.append({
            "acronym": acronym,
            "positive_hits": s["positive_hits"],
            "avg_synonyms": avg_syn
        })

    with open(out_dir / "mediating_ontology_ranking.json", "w") as f:
        json.dump(results, f, indent=2)

    # --------------------------------------------------------
    # Download top mediating ontology
    # --------------------------------------------------------

    if results:
        top_acronym = results[0]["acronym"]
        out_path = out_dir / f"{top_acronym}.owl"

        print(f"Downloading mediating ontology: {top_acronym}")

        bioportal_download_latest_submission(
            top_acronym,
            str(out_path)
        )

        print(f"Saved to {out_path}")