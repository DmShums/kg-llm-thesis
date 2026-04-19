from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime
import json
from tqdm import tqdm

from utils.prompts.logical_repair_prompt import (
    parse_logmap_repair_output,
    repair_plan_prompt,
    preprocess_reduced_prompt,
    select_plan,
    build_plan_selection_prompt
)
from utils.llm_server.qwen import QwenServer
from modules.llm_validator import LLMValidator
from utils.onto_object import OntologyAccess, OntologyEntryAttr


class LogicalRepairer:
    """
    High-level interface for:
    - Loading ontologies
    - Parsing LogMap repair plans
    - Building semantic prompts
    - Ranking repair plans using LLM
    """

    # ------------------------------------------------------------------
    # INIT
    # ------------------------------------------------------------------

    def __init__(
        self,
        o1_path: str,
        o2_path: str,
        llm_server: Optional[QwenServer] = None,
        use_prebuilt_prompts: bool = False,
        model: str = "qwen/qwen3-vl-8b-instruct"
    ):
        self.o1_path = o1_path
        self.o2_path = o2_path
        self.model = model

        if not use_prebuilt_prompts:
            self.onto_src = OntologyAccess(o1_path, annotate_on_init=True)
            self.onto_tgt = OntologyAccess(o2_path, annotate_on_init=True)

        self.llm = llm_server or QwenServer()

        # cache for prebuilt prompts
        self._prebuilt_cache: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # PARSING
    # ------------------------------------------------------------------

    def load_repair_plans(self, logmap_output_path: str) -> List[Dict]:
        """Parse LogMap repair output."""
        return parse_logmap_repair_output(
            logmap_output_path=logmap_output_path,
            o1_path=self.o1_path,
            o2_path=self.o2_path,
        )

    # ------------------------------------------------------------------
    # PROMPT BUILDING
    # ------------------------------------------------------------------

    def build_prompt(self, conflict: Dict) -> Optional[str]:
        """Build ranking prompt for one conflict."""
        conflict_entity: OntologyEntryAttr = conflict.get("entity_obj")

        if conflict_entity is None:
            return None

        return repair_plan_prompt(
            conflict_entity=conflict_entity,
            repair_plans=conflict["plans"],
        )

    # ------------------------------------------------------------------
    # PREBUILT PROMPTS
    # ------------------------------------------------------------------

    def save_prebuilt_prompts(
        self,
        logmap_output_path: str,
        dataset_name: str,
        experiment_type: str,
        limit: Optional[int] = None,
        output_dir: str = "data/LogMap_repair_plans/prebuilt_prompts",
    ) -> Dict[str, str]:

        conflicts = self.load_repair_plans(logmap_output_path)

        if limit:
            conflicts = conflicts[:limit]

        prompts: Dict[str, str] = {}

        for conflict in conflicts:
            prompt = self.build_prompt(conflict)
            if prompt:
                entity_uri = conflict.get("entity_uri")
                prompts[entity_uri] = prompt

        output_path = Path(output_dir) / dataset_name / experiment_type
        output_path.mkdir(parents=True, exist_ok=True)

        file_path = output_path / f"prebuilt_{dataset_name}.json"

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(prompts, f, indent=4, ensure_ascii=False)

        return prompts

    def load_prebuilt_prompts(
        self,
        dataset_name: str,
        experiment_type: str,
        base_dir: str = "data/LogMap_repair_plans/prebuilt_prompts",
    ) -> List[Dict]:

        file_path = (
            Path(base_dir)
            / dataset_name
            / experiment_type
            / f"prebuilt_{dataset_name}.json"
        )

        if not file_path.exists():
            raise FileNotFoundError(f"Prebuilt prompts not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            self._prebuilt_cache = json.load(f)

        conflicts = [
            {"entity_uri": uri, "plans": None}
            for uri in self._prebuilt_cache.keys()
        ]

        return conflicts

    def get_prebuilt_prompt(
        self,
        dataset_name: str,
        experiment_type: str,
        entity_uri: str,
    ) -> Optional[str]:
        """Retrieve cached prebuilt prompt."""
        return self._prebuilt_cache.get(entity_uri)

    # ------------------------------------------------------------------
    # SAVE RESULTS
    # ------------------------------------------------------------------

    def _save_results(
        self,
        results: List[Dict],
        dataset_name: str,
        experiment_type: str,
        results_dir: str = "output/repair_results",
        prompts: Optional[List[str]] = None,
    ) -> Path:

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

        results_path = (
            Path(results_dir)
            / f"repair_{timestamp}"
            / dataset_name
            / experiment_type
        )

        results_path.mkdir(parents=True, exist_ok=True)

        with open(results_path / "repair_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        if prompts:
            with open(results_path / "repair_prompts.txt", "w", encoding="utf-8") as f:
                f.write("\n\n".join(prompts))

        return results_path

    # ------------------------------------------------------------------
    # FULL PIPELINE: Includes LogMap heuristic + LLM-based ranking
    # ------------------------------------------------------------------
    def run(
        self,
        logmap_output_path: str,
        dataset_name: str,
        experiment_type: str,
        limit: Optional[int] = None,
        save: bool = True,
        use_prebuilt: bool = False,
    ) -> List[Dict]:

        # Load data
        if use_prebuilt:
            conflicts = self.load_prebuilt_prompts(
                dataset_name,
                experiment_type,
            )
        else:
            conflicts = self.load_repair_plans(logmap_output_path)

        if limit:
            conflicts = conflicts[:limit]

        results: List[Dict] = []
        used_prompts: List[str] = []

        # Main loop
        for conflict in conflicts:

            if use_prebuilt:
                prompt = self.get_prebuilt_prompt(
                    dataset_name,
                    experiment_type,
                    conflict.get("entity_uri"),
                )
            else:
                prompt = self.build_prompt(conflict)

            if not prompt:
                continue

            response = self.llm.ask_repair_ranking(prompt)

            results.append(
                {
                    "entity_uri": conflict.get("entity_uri"),
                    "plans": conflict.get("plans"),
                    "llm_response": response.message,
                    "parsed": (
                        response.parsed.model_dump()
                        if response.parsed else None
                    ),
                    "usage": response.usage.model_dump(),
                }
            )

            used_prompts.append(prompt)

        if save:
            self._save_results(
                results,
                dataset_name,
                experiment_type,
                prompts=used_prompts,
            )

        return results

    # ------------------------------------------------------------------
    # REDUCED PIPELINE: Includes LLM-based ranking without LogMap heuristics
    # ------------------------------------------------------------------
    def run_reduced(
        self,
        logmap_output_path: str,
        dataset_name: str,
        experiment_type: str,
        limit: Optional[int] = None,
        save: bool = True,
    ) -> List[Dict]:

        conflicts = self.load_repair_plans(logmap_output_path)

        if limit:
            conflicts = conflicts[:limit]

        validator = LLMValidator(llm_server=self.llm)

        results: List[Dict] = []

        for conflict in tqdm(conflicts, desc="Processing conflicts"):

            conflict_entity: OntologyEntryAttr = conflict["entity_obj"]

            plans = preprocess_reduced_prompt(
                conflict_entity=conflict_entity,
                repair_plans=conflict["plans"],
                onto_src=self.onto_src,
                onto_tgt=self.onto_tgt
            )

            evaluated_plans = []

            # ------------------------------------------------
            # Step 1 — validate mappings inside each plan
            # ------------------------------------------------

            for plan in plans:

                evaluated_mappings = []
                plan_valid = True

                for mapping in plan["mappings"]:


                    if mapping["decision"] == 0:
                        prompt_type = "source_subsumed_by_target"
                    elif mapping["decision"] == -1:
                        prompt_type = "target_subsumed_by_source"

                    validation = validator.validate(
                            mapping["src_entity"],
                            mapping["tgt_entity"],
                            prompt_type=prompt_type,
                            model=self.model
                        )

                    evaluated_mappings.append({
                        "source_uri": mapping["source_uri"],
                        "target_uri": mapping["target_uri"],
                        "src_entity_name": next(iter(mapping["src_entity"].get_preffered_names()), None),
                        "tgt_entity_name": next(iter(mapping["tgt_entity"].get_preffered_names()), None),
                        "decision": mapping["decision"],
                        "validation": validation,
                    })

                    if not validation["is_match"]:
                        plan_valid = False

                evaluated_plans.append({
                    "plan_id": plan["plan_id"],
                    "size": plan["size"],
                    "conflict_score": plan["conflict_score"],
                    "confidence": plan["confidence"],
                    "plan_valid": plan_valid,
                    "mappings": evaluated_mappings,
                })

            # ------------------------------------------------
            # Step 2 — automatic selection
            # ------------------------------------------------

            false_plans = [p for p in evaluated_plans if not p["plan_valid"]]

            if len(false_plans) == 1:

                selected_plan = false_plans[0]["plan_id"]
                reasoning = None
                decision_type = "single_invalid"

            else:

                arbitration_prompt = build_plan_selection_prompt(
                    evaluated_plans,
                    onto_src=self.onto_src,
                    onto_tgt=self.onto_tgt
                )

                response = self.llm.ask_repair_ranking(arbitration_prompt, model=self.model)
                raw = response.message
                try:
                    parsed_data = json.loads(raw)
                except json.JSONDecodeError:
                    # Attempt to extract JSON manually
                    start = raw.find("{")
                    end = raw.rfind("}") + 1
                    safe_json = raw[start:end]
                    safe_json = safe_json.replace('\n', '\\n').replace('“', '"').replace('”', '"')
                    parsed_data = json.loads(safe_json)

                selected_plan = parsed_data["selected_plan"]
                reasoning = parsed_data.get("reasoning")
                decision_type = "llm_reasoning"

            # ------------------------------------------------
            # Step 3 — save result
            # ------------------------------------------------

            results.append({
                "entity_uri": conflict["entity_uri"],
                "selected_plan": selected_plan,
                "decision_type": decision_type,
                "reasoning": reasoning,
                "plans": evaluated_plans,
            })

        if save:
            self._save_results(
                results,
                dataset_name,
                experiment_type
            )

        return results
    
