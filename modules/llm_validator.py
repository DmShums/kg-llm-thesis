import requests
from typing import Optional, Dict, Any, Callable

from utils.constants import LOGGER
from utils.onto_object import OntologyEntryAttr
from prompts.prompts import (
    prompt_direct_entity,
    prompt_direct_entity_ontological,
    prompt_direct_entity_with_synonyms,
    prompt_sequential_hierarchy,
    prompt_sequential_hierarchy_ontological,
    prompt_sequential_hierarchy_with_synonyms,
)
from prompts.system import SYSPROMPTS_MAP


PROMPT_FUNCTIONS: Dict[str, Callable] = {
    "direct_entity": prompt_direct_entity,
    "direct_entity_ontological": prompt_direct_entity_ontological,
    "direct_entity_with_synonyms": prompt_direct_entity_with_synonyms,
    "sequential_hierarchy": prompt_sequential_hierarchy,
    "sequential_hierarchy_ontological": prompt_sequential_hierarchy_ontological,
    "sequential_hierarchy_with_synonyms": prompt_sequential_hierarchy_with_synonyms,
}

# LLM Servers
from utils.llm_server.qwen import QwenServer

class LLMValidator:
    """
    LLM-based ontology mapping validator using Qwen server.
    """

    def __init__(
        self,
        server_url: str = "http://localhost:8000/chat",
        default_prompt: str = "direct_entity_with_synonyms",
        default_system_prompt: str = "ontology_aware",
        max_new_tokens: int = 50,
        temperature: float = 0.0,
    ):
        self.server_url = server_url
        self.default_prompt = default_prompt
        self.default_system_prompt = default_system_prompt
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        LOGGER.info("LLMValidator initialized")

    # ------------------------------------------------------------
    # Qwen call
    # ------------------------------------------------------------

    def _call_qwen(self, messages: list[dict]) -> str:
        payload = {
            "messages": messages,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
        }

        response = requests.post(self.server_url, json=payload, timeout=120)
        response.raise_for_status()

        raw = response.json()["response"].strip()
        LOGGER.debug(f"Qwen raw response: {raw}")
        return raw

    # ------------------------------------------------------------
    # Prompt builder
    # ------------------------------------------------------------

    def _build_messages(
        self,
        src_entity: OntologyEntryAttr,
        tgt_entity: OntologyEntryAttr,
        prompt_type: Optional[str],
        system_prompt_type: Optional[str],
    ) -> list[dict]:

        prompt_type = prompt_type or self.default_prompt
        system_prompt_type = system_prompt_type or self.default_system_prompt

        prompt_func = PROMPT_FUNCTIONS.get(prompt_type)
        if not prompt_func:
            raise ValueError(f"Unknown prompt_type: {prompt_type}")

        user_prompt = prompt_func(src_entity, tgt_entity)

        messages = []

        system_msg = SYSPROMPTS_MAP.get(system_prompt_type)
        if system_msg:
            messages.append({"role": "system", "content": system_msg})

        messages.append({"role": "user", "content": user_prompt})

        return messages

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------

    def validate(
        self,
        src_entity: OntologyEntryAttr,
        tgt_entity: OntologyEntryAttr,
        prompt_type: Optional[str] = None,
        system_prompt_type: Optional[str] = None,
        model: Optional[str] = "qwen",
    ) -> Dict[str, Any]:

        messages = self._build_messages(
            src_entity,
            tgt_entity,
            prompt_type,
            system_prompt_type,
        )

        if model.startswith("qwen"):
            llm_server = QwenServer()
            raw_response = llm_server.ask_sync_question(message=messages)
        else:
            raise ValueError(f"Unknown model: {model}")
        
        raw_response = self._call_qwen(messages)
        decision = raw_response.strip().lower()

        is_match = decision.startswith("true")

        result = {
            "is_match": is_match,
            "mapping_type": "EXACT_MATCH" if is_match else "NO_MATCH",
            "confidence": 0.95 if is_match else 0.05,
            "raw_response": raw_response,
            "prompt_type": prompt_type or self.default_prompt,
            "system_prompt_type": system_prompt_type or self.default_system_prompt,
        }

        LOGGER.info(
            f"Validation result: {result['mapping_type']} "
            f"(prompt={result['prompt_type']}, system={result['system_prompt_type']})"
        )

        return result
