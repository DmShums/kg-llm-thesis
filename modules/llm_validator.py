from locale import normalize
from pyexpat.errors import messages
from random import random
from typing import Optional, Dict, Any, Callable, List
from utils.constants import LOGGER, LLMCallOutput
from utils.onto_object import OntologyEntryAttr
from utils.prompts.prompts import (
    prompt_direct_entity,
    prompt_direct_entity_ontological,
    prompt_direct_entity_with_synonyms,
    prompt_sequential_hierarchy,
    prompt_sequential_hierarchy_ontological,
    prompt_sequential_hierarchy_with_synonyms,
    prompt_direct_entity_children,
    prompt_direct_entity_children_no_parents,
    prompt_source_subsumed_by_target,
    prompt_target_subsumed_by_source
)

from utils.prompts.system import SYSPROMPTS_MAP
from utils.llm_server.open_router import OpenRouterServer
import os

import random

random.seed(42)

# -------------------------------
# Mapping prompt types to functions
# -------------------------------

PROMPT_FUNCTIONS: Dict[str, Callable[[OntologyEntryAttr, OntologyEntryAttr], str]] = {
    "direct_entity": prompt_direct_entity,
    "direct_entity_ontological": prompt_direct_entity_ontological,
    "direct_entity_with_synonyms": prompt_direct_entity_with_synonyms,
    "sequential_hierarchy": prompt_sequential_hierarchy,
    "sequential_hierarchy_ontological": prompt_sequential_hierarchy_ontological,
    "sequential_hierarchy_with_synonyms": prompt_sequential_hierarchy_with_synonyms,
    "prompt_direct_entity_children": prompt_direct_entity_children,
    "prompt_direct_entity_children_no_parents": prompt_direct_entity_children_no_parents,
    "source_subsumed_by_target": prompt_source_subsumed_by_target,
    "target_subsumed_by_source": prompt_target_subsumed_by_source,
}


# -------------------------------
# LLM Validator
# -------------------------------

class LLMValidator:

    def __init__(
        self,
        llm_server: Optional[OpenRouterServer] = None,
        default_prompt: str = "direct_entity_with_synonyms",
        default_system_prompt: str = "ontology_aware",
        max_new_tokens: int = 50,
        temperature: float = 0.0,
    ):
        self.llm_server = llm_server or OpenRouterServer(api_key=os.environ.get("OPENROUTER_API_KEY"))
        self.default_prompt = default_prompt
        self.default_system_prompt = default_system_prompt
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        LOGGER.info("LLMValidator initialized with OpenRouterServer")


    # -------------------------------
    # Build messages
    # -------------------------------

    def _build_messages(
        self,
        src_entity: OntologyEntryAttr,
        tgt_entity: OntologyEntryAttr,
        prompt_type: Optional[str] = None,
        system_prompt_type: Optional[str] = None,
    ) -> List[Dict[str, str]]:

        prompt_type = prompt_type or self.default_prompt
        system_prompt_type = system_prompt_type or self.default_system_prompt

        prompt_func = PROMPT_FUNCTIONS.get(prompt_type)
        if prompt_func is None:
            raise ValueError(f"Unknown prompt_type: {prompt_type}")

        user_prompt = prompt_func(src_entity, tgt_entity)

        messages: List[Dict[str, str]] = []

        system_msg = SYSPROMPTS_MAP.get(system_prompt_type)
        if system_msg:
            messages.append({"role": "system", "content": system_msg})

        messages.append({"role": "user", "content": user_prompt})

        return messages


    # -------------------------------
    # Zero-shot validation
    # -------------------------------

    def validate(
        self,
        src_entity: OntologyEntryAttr,
        tgt_entity: OntologyEntryAttr,
        prompt_type: Optional[str] = None,
        system_prompt_type: Optional[str] = None,
        model: str = "qwen/qwen3-vl-8b-instruct",
    ) -> Dict[str, Any]:

        messages = self._build_messages(
            src_entity,
            tgt_entity,
            prompt_type,
            system_prompt_type
        )
        # print(messages)

        try:
            llm_output: LLMCallOutput = self.llm_server.ask_chat(
                messages=messages,
                model=model,
            )
            is_match = getattr(llm_output.parsed, "answer", False)
            confidence = getattr(llm_output.parsed, "confidence", None)
            raw_response = llm_output.message.strip()

            # Add token usage
            input_tokens = getattr(llm_output.usage, "input_tokens", 0)
            output_tokens = getattr(llm_output.usage, "output_tokens", 0)
            token_usage = input_tokens + output_tokens

        except RuntimeError as e:
            # Optional fallback: log the entity pair and continue
            print(f"⚠️ LLM call failed for pair: {src_entity.annotation} | {tgt_entity.annotation}")
            print(f"Reason: {e}")

            is_match = False
            confidence = None
            raw_response = ""
            token_usage = None

        result = {
            "is_match": is_match,
            "mapping_type": "EXACT_MATCH" if is_match else "NO_MATCH",
            "confidence": confidence,
            "raw_response": raw_response,
            "prompt_type": prompt_type or self.default_prompt,
            "system_prompt_type": system_prompt_type or self.default_system_prompt,
            "tokens_used": token_usage,
        }

        LOGGER.info(
            f"Validation result: {result['mapping_type']} "
            f"(prompt={result['prompt_type']}, system={result['system_prompt_type']})"
        )

        return result


    # -------------------------------
    # Few-shot validation
    # -------------------------------

    def validate_few_shot(
        self,
        src_entity: OntologyEntryAttr,
        tgt_entity: OntologyEntryAttr,
        few_shot_examples: List[Dict[str, str]],
        k: int = 3,
        prompt_type: Optional[str] = None,
        system_prompt_type: Optional[str] = None,
        model: str = "qwen/qwen3-vl-8b-instruct",
    ) -> Dict[str, Any]:

        if len(few_shot_examples) == 0:
            raise ValueError("few_shot_examples is empty")

        prompt_type = prompt_type or self.default_prompt
        system_prompt_type = system_prompt_type or self.default_system_prompt

        prompt_func = PROMPT_FUNCTIONS.get(prompt_type)
        if prompt_func is None:
            raise ValueError(f"Unknown prompt_type: {prompt_type}")

        query_prompt = prompt_func(src_entity, tgt_entity)

        messages: List[Dict[str, str]] = []

        system_msg = SYSPROMPTS_MAP.get(system_prompt_type)
        if system_msg:
            messages.append({"role": "system", "content": system_msg})

        selected_examples = few_shot_examples[:min(k, len(few_shot_examples))]

        for ex in selected_examples:

            messages.append({
                "role": "user",
                "content": ex["input"]
            })

            messages.append({
                "role": "assistant",
                "content": ex["output"]
            })

        messages.append({
            "role": "user",
            "content": query_prompt
        })

        # print(messages)

        try:
            llm_output = self.llm_server.ask_chat(
                messages=messages,
                model=model,
            )
            is_match = getattr(llm_output.parsed, "answer", False)
            confidence = getattr(llm_output.parsed, "confidence", None)
            raw_response = llm_output.message.strip()

            # Add token usage
            input_tokens = getattr(llm_output.usage, "input_tokens", 0)
            output_tokens = getattr(llm_output.usage, "output_tokens", 0)
            token_usage = input_tokens + output_tokens

        except RuntimeError as e:
            print(f"⚠️ LLM call failed for pair: {src_entity.annotation} | {tgt_entity.annotation}")
            print(f"Reason: {e}")

            is_match = False
            confidence = None
            raw_response = ""
            token_usage = None  

        result = {
            "is_match": is_match,
            "mapping_type": "EXACT_MATCH" if is_match else "NO_MATCH",
            "confidence": confidence,
            "raw_response": raw_response,
            "prompt_type": prompt_type,
            "system_prompt_type": system_prompt_type,
            "num_examples": len(selected_examples),
            "tokens_used": token_usage,
        }

        LOGGER.info(
            f"Few-shot validation: {result['mapping_type']} "
            f"(examples={len(selected_examples)})"
        )

        return result


    # -------------------------------
    # Few-shot RAG validation
    # -------------------------------

    def validate_few_shot_rag(
        self,
        src_entity: OntologyEntryAttr,
        tgt_entity: OntologyEntryAttr,
        vectorstore,
        few_shot_examples: List[Dict[str, str]],
        k: int = 3,
        prompt_type: Optional[str] = None,
        system_prompt_type: Optional[str] = None,
        model: str = "qwen/qwen3-vl-8b-instruct",
    ) -> Dict[str, Any]:

        prompt_type = prompt_type or self.default_prompt
        system_prompt_type = system_prompt_type or self.default_system_prompt

        prompt_func = PROMPT_FUNCTIONS.get(prompt_type)
        if prompt_func is None:
            raise ValueError(f"Unknown prompt_type: {prompt_type}")

        query_prompt = prompt_func(src_entity, tgt_entity)

        messages: List[Dict[str, str]] = []

        system_msg = SYSPROMPTS_MAP.get(system_prompt_type)
        if system_msg:
            messages.append({"role": "system", "content": system_msg})

        def normalize(x: str) -> str:
            if not x:
                return ""
            return (
                x.replace("_", " ")
                .replace("-", " ")
                .lower()
                .strip()
            )

        def get_first_name(names):
            if not names:
                return ""
            first = next(iter(names))
            if isinstance(first, set):
                return next(iter(first))
            return str(first)

        src_label = normalize(next(iter(src_entity.get_preffered_names())))
        tgt_label = normalize(next(iter(tgt_entity.get_preffered_names())))

        src_parent = ""
        tgt_parent = ""

        if src_entity.get_direct_parents():
            src_parent = normalize(get_first_name(src_entity.get_parents_preferred_names()))

        if tgt_entity.get_direct_parents():
            tgt_parent = normalize(get_first_name(tgt_entity.get_parents_preferred_names()))

        # Pair retrieval query
        pair_query = f"query: {src_label} {src_parent} {tgt_label} {tgt_parent}"

        pool_size = max(30, k * 10)

        results = vectorstore.similarity_search_with_score(pair_query, k=pool_size)

        retrieved_docs = []
        for doc, score in results:

            src_meta = doc.metadata.get("src", "").lower()
            tgt_meta = doc.metadata.get("tgt", "").lower()

            if src_meta == src_label and tgt_meta == tgt_label:
                continue

            retrieved_docs.append((doc, score))

        # sort by similarity
        retrieved_docs.sort(key=lambda x: x[1])

        positives = []
        negatives = []

        for doc, score in retrieved_docs:

            label = doc.metadata.get("label", "").lower()

            if "true" in label:
                positives.append(doc)
            else:
                negatives.append(doc)

        selected_docs = []

        pos_needed = (k + 1) // 2
        neg_needed = k // 2

        p = positives[:pos_needed]
        n = negatives[:neg_needed]

        selected_docs = []

        for i in range(k):
            if i % 2 == 0:  # even index - positive
                if p:
                    selected_docs.append(p.pop(0))
                elif n:
                    selected_docs.append(n.pop(0))
            else:  # odd index - negative
                if n:
                    selected_docs.append(n.pop(0))
                elif p:
                    selected_docs.append(p.pop(0))

        selected_docs = selected_docs[:k]

        selected_examples = []
        used_idx = set()

        for doc in selected_docs:

            idx = doc.metadata["index"]

            if idx in used_idx:
                continue

            used_idx.add(idx)

            if idx < len(few_shot_examples):
                selected_examples.append(few_shot_examples[idx])

        for ex in selected_examples:

            messages.append({
                "role": "user",
                "content": ex["input"]
            })

            messages.append({
                "role": "assistant",
                "content": ex["output"]
            })

        messages.append({
            "role": "user",
            "content": query_prompt
        })

        try:
            print("* Messages: ", messages)
            llm_output = self.llm_server.ask_chat(
                messages=messages,
                model=model,
            )

            is_match = getattr(llm_output.parsed, "answer", False)
            confidence = getattr(llm_output.parsed, "confidence", None)
            raw_response = llm_output.message.strip()

            input_tokens = getattr(llm_output.usage, "input_tokens", 0)
            output_tokens = getattr(llm_output.usage, "output_tokens", 0)
            token_usage = input_tokens + output_tokens

        except RuntimeError as e:

            print(f"⚠️ LLM call failed for pair: {src_entity.annotation} | {tgt_entity.annotation}")
            print(f"Reason: {e}")

            is_match = False
            confidence = None
            raw_response = ""
            token_usage = None

        result = {
            "is_match": is_match,
            "mapping_type": "EXACT_MATCH" if is_match else "NO_MATCH",
            "confidence": confidence,
            "raw_response": raw_response,
            "prompt_type": prompt_type,
            "system_prompt_type": system_prompt_type,
            "num_examples": len(selected_examples),
            "retrieval": "pair_similarity",
            "tokens_used": token_usage,
        }

        LOGGER.info(
            f"Few-shot RAG validation: {result['mapping_type']} "
            f"(retrieved={len(selected_examples)})"
        )

        return result