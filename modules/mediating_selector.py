# ------------------------------------------------------------
# mediating_selector.py
# ------------------------------------------------------------

import json
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Set

from utils.onto_access import OntologyAccess
from utils.rag import bioportal_search, bioportal_download_latest_submission
from utils.constants import LOGGER


class MediatingOntologySelector:
    def __init__(
        self,
        o1_path: str,
        o2_path: str,
        lexical_mappings: List[Tuple[str, str]] = None,
        cache_dir: str = "output/mediators",
        json_out: str = "output/mediators/mediating_ontology_ranking.json",
        stability_calls: int = 25,
    ):
        self.o1 = OntologyAccess(o1_path)
        self.o2 = OntologyAccess(o2_path)
        self.lexical_mappings = lexical_mappings or []
        self.stability_calls = stability_calls

        # Build IRI → label index
        self.o1_labels = self._build_label_index(self.o1)
        self.o2_labels = self._build_label_index(self.o2)

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.json_out = Path(json_out)
        self.json_out.parent.mkdir(parents=True, exist_ok=True)

        LOGGER.info("MediatingOntologySelector initialized")

    # ------------------------------------------------------------
    # Load LogMap anchor mappings
    # ------------------------------------------------------------
    @staticmethod
    def load_logmap_anchors(path: str) -> List[Tuple[str, str]]:
        """
        Load LogMap anchor mappings from a TSV file.
        Each line should contain two IRIs separated by a tab.
        Lines with fewer than 2 columns are ignored.
        """
        mappings: List[Tuple[str, str]] = []

        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                iri1, iri2 = parts[0], parts[1]
                mappings.append((iri1, iri2))

        LOGGER.info(f"Loaded {len(mappings)} lexical mappings from {path}")
        return mappings

    # ------------------------------------------------------------
    # Build label index
    # ------------------------------------------------------------
    @staticmethod
    def _build_label_index(onto: OntologyAccess) -> Dict[str, str]:
        index = {}
        for cls in onto.getClasses():
            label = getattr(cls, "label", None)
            if not label:
                continue
            if isinstance(label, list):
                label = label[0]
            index[str(cls.iri)] = str(label).lower()
        return index

    # ------------------------------------------------------------
    # Extract labels from lexical mappings
    # ------------------------------------------------------------
    def extract_labels_from_mappings(self) -> List[str]:
        LOGGER.info("Extracting labels from lexical mappings")

        labels: Set[str] = set()
        for iri1, iri2 in self.lexical_mappings:
            l1 = self.o1_labels.get(iri1)
            l2 = self.o2_labels.get(iri2)
            if l1:
                labels.add(l1)
            if l2:
                labels.add(l2)

        LOGGER.info(f"Collected {len(labels)} representative labels")
        return list(labels)

    # ------------------------------------------------------------
    # BioPortal helpers
    # ------------------------------------------------------------
    @staticmethod
    def _extract_acronym(result: dict) -> Optional[str]:
        onto_url = result.get("links", {}).get("ontology")
        if not onto_url:
            return None
        return onto_url.rstrip("/").split("/")[-1]

    # ------------------------------------------------------------
    # Mediating candidates collection
    # ------------------------------------------------------------
    def collect_mediating_candidates(self, labels: List[str]) -> Dict[str, dict]:
        LOGGER.info("Querying BioPortal for mediating candidates")
        stats = defaultdict(lambda: {"positive_hits": 0, "synonym_count": 0})

        stable = 0
        prev_size = 0

        for label in tqdm(labels, desc="Processing labels"):
            results = bioportal_search(label)
            seen_ontologies = set()

            for r in results:
                onto = self._extract_acronym(r)
                synonyms = r.get("synonym", [])
                if onto and synonyms and onto not in seen_ontologies:
                    stats[onto]["positive_hits"] += 1
                    stats[onto]["synonym_count"] += len(synonyms)
                    seen_ontologies.add(onto)

            if len(stats) == prev_size:
                stable += 1
            else:
                stable = 0
                prev_size = len(stats)

            if stable >= self.stability_calls:
                LOGGER.info(f"Stop condition reached after {self.stability_calls} stable calls")
                break

        LOGGER.info(f"Collected stats for {len(stats)} candidate ontologies")
        return stats

    # ------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------
    @staticmethod
    def rank_mediators(stats: Dict[str, dict], top_k: int = 5) -> List[Tuple[str, dict]]:
        LOGGER.info("Ranking mediating ontologies")
        ranked = sorted(
            stats.items(),
            key=lambda x: (
                x[1]["positive_hits"],
                x[1]["synonym_count"] / max(x[1]["positive_hits"], 1),
            ),
            reverse=True,
        )
        return ranked[:top_k]

    # ------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------
    def _save_ranking_to_json(self, ranked: List[Tuple[str, dict]]):
        LOGGER.info(f"Saving ranking to {self.json_out}")
        results = [
            {
                "acronym": acronym,
                "positive_hits": s["positive_hits"],
                "avg_synonyms": s["synonym_count"] / max(s["positive_hits"], 1),
            }
            for acronym, s in ranked
        ]
        with open(self.json_out, "w") as f:
            json.dump(results, f, indent=2)

    # ------------------------------------------------------------
    # Download ontology
    # ------------------------------------------------------------
    def get_or_download_ontology(self, acronym: str) -> Path:
        out_path = self.cache_dir / f"{acronym}.owl"
        if out_path.exists():
            LOGGER.info(f"Using cached ontology: {acronym}")
        else:
            LOGGER.info(f"Downloading mediating ontology: {acronym}")
            bioportal_download_latest_submission(acronym, str(out_path))
        return out_path

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------
    def select_top_mediators(self, top_k: int = 5, download: bool = False) -> List[dict]:
        LOGGER.info("Starting mediating ontology selection pipeline")

        labels = self.extract_labels_from_mappings()
        stats = self.collect_mediating_candidates(labels)
        ranked = self.rank_mediators(stats, top_k=top_k)
        self._save_ranking_to_json(ranked)

        results = []
        for acronym, s in ranked:
            entry = {
                "acronym": acronym,
                "positive_hits": s["positive_hits"],
                "avg_synonyms": s["synonym_count"] / max(s["positive_hits"], 1),
            }
            if download:
                try:
                    entry["local_path"] = str(self.get_or_download_ontology(acronym))
                except Exception as e:
                    # Note: Some ontologies may fail to download due to various reasons (API issues, ontology removed, license restrictions etc.)
                    LOGGER.warning(f"Failed to download ontology {acronym}: {e}")
            results.append(entry)

        LOGGER.info("Mediating ontology selection completed")
        return results