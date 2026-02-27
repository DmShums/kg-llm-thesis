from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime
import json

from prompts.logical_repair_prompt import (
    parse_logmap_repair_output,
    repair_plan_prompt
)
from utils.llm_server.qwen import QwenServer
from utils.onto_object import OntologyAccess, OntologyEntryAttr


class LogicalRepairer:
    """
    High-level interface for:
    - Loading ontologies
    - Parsing LogMap repair plans
    - Building semantic prompts
    - Ranking repair plans using LLM
    """

    def __init__(
        self,
        o1_path: str,
        o2_path: str,
        llm_server: Optional[QwenServer] = None
    ):
        self.o1_path = o1_path
        self.o2_path = o2_path

        # Load ontologies once
        self.onto_src = OntologyAccess(o1_path, annotate_on_init=True)
        self.onto_tgt = OntologyAccess(o2_path, annotate_on_init=True)

        # LLM (injectable for experiments)
        self.llm = llm_server or QwenServer()

    # ------------------------------------------------------------------
    # PARSING
    # ------------------------------------------------------------------

    def load_repair_plans(self, logmap_output_path: str) -> List[Dict]:
        """
        Parse LogMap repair output and enrich with OntologyEntryAttr.
        """
        return parse_logmap_repair_output(
            logmap_output_path=logmap_output_path,
            o1_path=self.o1_path,
            o2_path=self.o2_path
        )

    # ------------------------------------------------------------------
    # PROMPT BUILDING
    # ------------------------------------------------------------------

    def build_prompt(self, conflict: Dict) -> Optional[str]:
        """
        Build ranking prompt for a single conflict.
        """
        conflict_entity: OntologyEntryAttr = conflict.get("entity_obj")

        if conflict_entity is None:
            return None

        return repair_plan_prompt(
            conflict_entity=conflict_entity,
            repair_plans=conflict["plans"]
        )

    # ------------------------------------------------------------------
    # LLM RANKING
    # ------------------------------------------------------------------

    def rank_conflict(self, conflict: Dict) -> Optional[str]:
        """
        Run LLM ranking for a single conflict.
        Returns raw JSON string from LLM.
        """
        prompt = self.build_prompt(conflict)

        if prompt is None:
            return None

        return self.llm.ask_sync_question(message=prompt)

    # ------------------------------------------------------------------
    # SAVE RESULTS
    # ------------------------------------------------------------------

    def _save_results(
        self,
        results: List[Dict],
        dataset_name: str,
        experiment_type: str,
        results_dir: str = "results",
        prompts: Optional[List[str]] = None
    ) -> Path:
        """
        Save repair experiment results as JSON files with timestamp.
        """

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

        results_path = (
            Path(results_dir)
            / f"repair_{timestamp}"
            / dataset_name
            / experiment_type
        )

        results_path.mkdir(parents=True, exist_ok=True)

        # Save full JSON results
        with open(results_path / "repair_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)

        # Save prompts separately (optional)
        if prompts:
            with open(results_path / "repair_prompts.txt", "w", encoding="utf-8") as f:
                for p in prompts:
                    f.write(p.strip() + "\n\n")

        return results_path

    # ------------------------------------------------------------------
    # FULL PIPELINE
    # ------------------------------------------------------------------

    def run(
        self,
        logmap_output_path: str,
        dataset_name: str,
        experiment_type: str,
        limit: Optional[int] = None,
        save: bool = True
    ) -> List[Dict]:

        conflicts = self.load_repair_plans(logmap_output_path)

        if limit:
            conflicts = conflicts[:limit]

        results = []
        used_prompts = []

        for conflict in conflicts:

            prompt = self.build_prompt(conflict)

            if prompt is None:
                continue

            response = self.llm.ask_sync_question(message=prompt)

            results.append({
                "entity_uri": conflict.get("entity_uri"),
                "plans": conflict.get("plans"),
                "llm_response": response
            })

            used_prompts.append(prompt)

        if save:
            self._save_results(
                results=results,
                dataset_name=dataset_name,
                experiment_type=experiment_type,
                prompts=used_prompts
            )

        return results