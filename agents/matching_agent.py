"""
MatchingAgent: Handles matching source concepts to target candidates using multiple strategies.
"""

import json
import os
from typing import List, Dict, Any, Optional, Tuple
from utils.rag import call_qwen_chat, make_prompt
from utils.onto_object import OntologyEntryAttr
from utils.onto_access import OntologyAccess
from prompts.prompts import (
    prompt_direct_entity,
    prompt_direct_entity_with_synonyms,
    prompt_sequential_hierarchy,
    prompt_sequential_hierarchy_with_synonyms,
    prompt_direct_entity_ontological,
    prompt_sequential_hierarchy_ontological
)
from prompts.system import SYSPROMPTS_MAP
from utils.constants import LOGGER


class MatchingAgent:
    """
    MatchingAgent performs matching using multiple strategies:
    - RAG-based matching (vector similarity + LLM)
    - Ontological matching (using hierarchy and synonyms)
    - Direct entity matching
    """

    def __init__(
        self,
        qwen_server_url: Optional[str] = None,
        matching_strategies: Optional[List[str]] = None,
        system_prompt_type: str = "ontology_aware"
    ):
        self.qwen_server_url = qwen_server_url or os.environ.get("QWEN_SERVER_URL")
        self.matching_strategies = matching_strategies or ["rag", "ontological"]
        self.system_prompt = SYSPROMPTS_MAP.get(system_prompt_type, SYSPROMPTS_MAP["ontology_aware"])

    def match_with_rag(
        self,
        source_label: str,
        source_def: str,
        candidates: List[Tuple[Dict[str, Any], float]],
        top_k: int = 8
    ) -> List[Dict[str, Any]]:
        """
        Match using RAG: vector similarity + LLM classification.
        
        Args:
            source_label: Source concept label
            source_def: Source concept definition
            candidates: List of (candidate_dict, similarity_score) tuples
            top_k: Number of top candidates to process
        
        Returns:
            List of match results with mapping_type, confidence, reason
        """
        if not candidates:
            return []
        
        # Take top-k candidates
        top_candidates = candidates[:top_k]
        
        # Build prompt and call LLM
        prompt = make_prompt(source_label, source_def or "", top_candidates)
        
        try:
            raw_output = call_qwen_chat(
                prompt,
                server_url=self.qwen_server_url,
                max_new_tokens=800,
                temperature=0.0
            )
            
            # Parse JSON output
            try:
                parsed = json.loads(raw_output)
            except json.JSONDecodeError:
                # Try to extract JSON from text
                import re
                m = re.search(r'\[.*\]', raw_output, flags=re.DOTALL) or re.search(r'\{.*\}', raw_output, flags=re.DOTALL)
                if m:
                    parsed = json.loads(m.group(0))
                else:
                    LOGGER.warning(f"Failed to parse RAG match output: {raw_output}")
                    return []
            
            # Normalize to list
            if isinstance(parsed, dict):
                parsed = [parsed]
            
            # Validate and enrich results
            results = []
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                
                # Ensure required fields
                result = {
                    "candidate_uri": item.get("candidate_uri"),
                    "candidate_label": item.get("candidate_label"),
                    "mapping_type": item.get("mapping_type", "NO_MATCH"),
                    "confidence": float(item.get("confidence", 0.0)),
                    "reason": item.get("reason", ""),
                    "strategy": "rag"
                }
                
                # Clamp confidence to [0, 1]
                result["confidence"] = max(0.0, min(1.0, result["confidence"]))
                
                results.append(result)
            
            LOGGER.info(f"RAG matching returned {len(results)} matches")
            return results
            
        except Exception as e:
            LOGGER.error(f"RAG matching failed: {e}")
            return []

    def match_with_ontological_prompt(
        self,
        source_entity: OntologyEntryAttr,
        target_entity: OntologyEntryAttr,
        prompt_type: str = "direct_entity_with_synonyms"
    ) -> Dict[str, Any]:
        """
        Match using ontological prompts (hierarchy-aware).
        
        Args:
            source_entity: Source ontology entity
            target_entity: Target ontology entity
            prompt_type: Type of prompt to use
        
        Returns:
            Match result with True/False and reason
        """
        prompt_funcs = {
            "direct_entity": prompt_direct_entity,
            "direct_entity_with_synonyms": prompt_direct_entity_with_synonyms,
            "sequential_hierarchy": prompt_sequential_hierarchy,
            "sequential_hierarchy_with_synonyms": prompt_sequential_hierarchy_with_synonyms,
            "direct_entity_ontological": prompt_direct_entity_ontological,
            "sequential_hierarchy_ontological": prompt_sequential_hierarchy_ontological
        }
        
        prompt_func = prompt_funcs.get(prompt_type, prompt_direct_entity_with_synonyms)
        prompt = prompt_func(source_entity, target_entity)
        
        # Add system prompt if available
        if self.system_prompt:
            full_prompt = f"{self.system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt
        
        try:
            response = call_qwen_chat(
                full_prompt,
                server_url=self.qwen_server_url,
                max_new_tokens=50,
                temperature=0.0
            )
            
            # Parse True/False response
            response_lower = response.strip().lower()
            is_match = "true" in response_lower and "false" not in response_lower[:10]
            
            result = {
                "is_match": is_match,
                "confidence": 0.9 if is_match else 0.1,
                "reason": response,
                "mapping_type": "EXACT_MATCH" if is_match else "NO_MATCH",
                "strategy": f"ontological_{prompt_type}"
            }
            
            return result
            
        except Exception as e:
            LOGGER.error(f"Ontological matching failed: {e}")
            return {
                "is_match": False,
                "confidence": 0.0,
                "reason": f"Error: {str(e)}",
                "mapping_type": "NO_MATCH",
                "strategy": f"ontological_{prompt_type}"
            }

    def match_concept(
        self,
        source_label: str,
        source_def: Optional[str],
        source_entity: Optional[OntologyEntryAttr],
        candidates: List[Tuple[Dict[str, Any], float]],
        target_ontology: Optional[OntologyAccess] = None
    ) -> List[Dict[str, Any]]:
        """
        Main matching method that combines multiple strategies.
        
        Args:
            source_label: Source concept label
            source_def: Source concept definition
            source_entity: Optional source ontology entity (for ontological matching)
            candidates: List of (candidate_dict, score) tuples
            target_ontology: Optional target ontology access (for ontological matching)
        
        Returns:
            List of match results
        """
        all_matches = []
        
        # RAG-based matching
        if "rag" in self.matching_strategies:
            rag_matches = self.match_with_rag(
                source_label,
                source_def or "",
                candidates
            )
            all_matches.extend(rag_matches)
        
        # Ontological matching (if entities are available)
        if "ontological" in self.matching_strategies and source_entity and target_ontology:
            for candidate_dict, score in candidates[:5]:  # Limit to top 5 for efficiency
                candidate_uri = candidate_dict.get("@id") or candidate_dict.get("id") or candidate_dict.get("iri")
                if not candidate_uri:
                    continue
                
                try:
                    target_entity = OntologyEntryAttr(
                        class_uri=candidate_uri,
                        onto=target_ontology
                    )
                    
                    # Try different prompt types
                    for prompt_type in ["direct_entity_with_synonyms", "sequential_hierarchy_with_synonyms"]:
                        match_result = self.match_with_ontological_prompt(
                            source_entity,
                            target_entity,
                            prompt_type=prompt_type
                        )
                        
                        # Add candidate info
                        match_result["candidate_uri"] = candidate_uri
                        match_result["candidate_label"] = candidate_dict.get("prefLabel") or candidate_dict.get("label")
                        match_result["vector_score"] = score
                        
                        all_matches.append(match_result)
                        
                except Exception as e:
                    LOGGER.warning(f"Failed to create target entity for {candidate_uri}: {e}")
                    continue
        
        # Aggregate and deduplicate matches
        return self._aggregate_matches(all_matches)

    def _aggregate_matches(self, matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Aggregate matches by candidate URI, combining scores from different strategies."""
        uri_to_matches: Dict[str, List[Dict[str, Any]]] = {}
        
        for match in matches:
            uri = match.get("candidate_uri")
            if not uri:
                continue
            
            if uri not in uri_to_matches:
                uri_to_matches[uri] = []
            uri_to_matches[uri].append(match)
        
        aggregated = []
        for uri, match_list in uri_to_matches.items():
            # Combine confidences (average or max)
            confidences = [m.get("confidence", 0.0) for m in match_list]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            max_confidence = max(confidences) if confidences else 0.0
            
            # Use the match with highest confidence as base
            best_match = max(match_list, key=lambda m: m.get("confidence", 0.0))
            
            aggregated_match = {
                "candidate_uri": uri,
                "candidate_label": best_match.get("candidate_label"),
                "mapping_type": best_match.get("mapping_type", "NO_MATCH"),
                "confidence": max_confidence,  # Use max for aggregated
                "avg_confidence": avg_confidence,
                "reason": best_match.get("reason", ""),
                "strategies": [m.get("strategy", "unknown") for m in match_list],
                "match_count": len(match_list)
            }
            
            aggregated.append(aggregated_match)
        
        # Sort by confidence descending
        aggregated.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)
        
        return aggregated
