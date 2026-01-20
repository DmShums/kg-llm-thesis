


LLM as planner, tools as actuators

Clear separation of roles — e.g. RetrievalAgent (search/embedding/DB), MatchingAgent (rank fusion + validation), MergerAgent (merge + metrics). That mirrors Agent-OM and makes each piece testable and replaceable.


# RetrievalAgent

Responsibilities: call BioPortal search/annotator, normalize candidate_to_text, build/update vector index (FAISS), cache metadata.

Tools: search_bioportal, annotate_bioportal, index_add, index_query, inspect_index.

# MatchingAgent

Responsibilities: given source concept, call retrieval tools (or query vector DB), create prompt, call Qwen (or local LLM), parse result, run RRF if you aggregate multiple retrievals (BioPortal + local KB + pgvector).

Tools: build_prompt, call_qwen_chat, parse_json, reciprocal_rank_fusion, validate_mapping.

# ValidationAgent (or Validator tool)

Responsibilities: call LLM to validate candidate pairs, perform strict heuristics (string equality, normalized tokens), fallback to LLM only for ambiguous cases.

# MergeAgent / Orchestrator

Responsibilities: merge source↔target matches, compute metrics, persist results, trigger audits.

(Optional) AuditorAgent / HumanAgent