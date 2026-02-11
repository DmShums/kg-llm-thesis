import json
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

from utils.onto_access import OntologyAccess
from utils.rag import bioportal_search, bioportal_download_latest_submission
from utils.constants import LOGGER


class MediatingOntologySelector:
    def __init__(
        self,
        o1_path: str,
        o2_path: str,
        cache_dir: str = "output/mediators",
        json_out: str = "output/mediators/mediating_ontology_ranking.json",
        max_queries: int = 25,
    ):
        self.o1 = OntologyAccess(o1_path)
        self.o2 = OntologyAccess(o2_path)
        self.max_queries = max_queries

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.json_out = Path(json_out)
        self.json_out.parent.mkdir(parents=True, exist_ok=True)

        LOGGER.info("MediatingOntologySelector initialized")

    # ------------------------------------------------------------
    # Step 1: Exact lexical overlap
    # ------------------------------------------------------------

    @staticmethod
    def _collect_labels(onto: OntologyAccess) -> set:
        labels = set()
        for cls in onto.getClasses():
            label = getattr(cls, "label", None)
            if label:
                if isinstance(label, list):
                    labels.add(str(label[0]).lower())
                else:
                    labels.add(str(label).lower())
        return labels

    def extract_exact_label_matches(self) -> List[str]:
        LOGGER.info("Extracting exact lexical overlaps between ontologies")

        labels_o1 = self._collect_labels(self.o1)
        labels_o2 = self._collect_labels(self.o2)
        shared = list(labels_o1 & labels_o2)

        LOGGER.info(f"Found {len(shared)} shared labels")
        return shared

    # ------------------------------------------------------------
    # BioPortal helpers
    # ------------------------------------------------------------

    @staticmethod
    def _extract_acronym(result: dict) -> Optional[str]:
        onto_url = result.get("links", {}).get("ontology")
        if not onto_url:
            return None
        return onto_url.rstrip("/").split("/")[-1]

    def collect_mediating_candidates(
        self,
        shared_labels: List[str]
    ) -> Dict[str, dict]:

        LOGGER.info("Querying BioPortal SEARCH for mediating candidates")

        stats = defaultdict(lambda: {
            "positive_hits": 0,
            "synonym_count": 0,
        })

        for i, label in enumerate(shared_labels):
            if i >= self.max_queries:
                LOGGER.info(f"Reached max_queries limit: {self.max_queries}")
                break

            LOGGER.debug(f"BioPortal search for label: '{label}'")

            results = bioportal_search(label)
            seen_ontologies = set()

            for r in results:
                onto = self._extract_acronym(r)
                synonyms = r.get("synonym", [])

                if onto and synonyms and onto not in seen_ontologies:
                    stats[onto]["positive_hits"] += 1
                    stats[onto]["synonym_count"] += len(synonyms)
                    seen_ontologies.add(onto)

        LOGGER.info(f"Collected stats for {len(stats)} candidate ontologies")
        return stats

    # ------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------

    @staticmethod
    def rank_mediators(
        stats: Dict[str, dict],
        top_k: int = 5
    ) -> List[Tuple[str, dict]]:

        LOGGER.info("Ranking mediating ontologies")

        ranked = sorted(
            stats.items(),
            key=lambda x: (
                x[1]["positive_hits"],
                x[1]["synonym_count"] / max(x[1]["positive_hits"], 1)
            ),
            reverse=True
        )

        return ranked[:top_k]

    # ------------------------------------------------------------
    # JSON trace (same style as your script)
    # ------------------------------------------------------------

    def _save_ranking_to_json(self, ranked: List[Tuple[str, dict]]):
        LOGGER.info(f"Saving mediating ontology ranking to {self.json_out}")

        results = []
        for acronym, s in ranked:
            avg_syn = s["synonym_count"] / max(s["positive_hits"], 1)
            results.append({
                "acronym": acronym,
                "positive_hits": s["positive_hits"],
                "avg_synonyms": avg_syn
            })

        with open(self.json_out, "w") as f:
            json.dump(results, f, indent=2)

    # ------------------------------------------------------------
    # Download logic (cached)
    # ------------------------------------------------------------

    def get_or_download_ontology(self, acronym: str) -> Path:
        out_path = self.cache_dir / f"{acronym}.owl"

        if out_path.exists():
            LOGGER.info(f"Using cached ontology: {acronym}")
        else:
            LOGGER.info(f"Downloading mediating ontology: {acronym}")
            bioportal_download_latest_submission(
                acronym,
                str(out_path)
            )

        return out_path

    # ------------------------------------------------------------
    # Public API for agent
    # ------------------------------------------------------------

    def select_top_mediators(
        self,
        top_k: int = 5,
        download: bool = False
    ) -> List[dict]:

        LOGGER.info("Starting mediating ontology selection pipeline")

        shared = self.extract_exact_label_matches()
        stats = self.collect_mediating_candidates(shared)
        ranked = self.rank_mediators(stats, top_k=top_k)

        self._save_ranking_to_json(ranked)

        results = []

        for acronym, s in ranked:
            avg_syn = s["synonym_count"] / max(s["positive_hits"], 1)

            entry = {
                "acronym": acronym,
                "positive_hits": s["positive_hits"],
                "avg_synonyms": avg_syn,
            }

            if download:
                entry["local_path"] = str(self.get_or_download_ontology(acronym))

            results.append(entry)

        LOGGER.info("Mediating ontology selection completed")
        return results
