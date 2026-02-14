# langchain_agent_rag.py
import os
import json
import time
from typing import List, Dict, Any, Tuple, Optional
import requests

# embeddings + faiss
from sentence_transformers import SentenceTransformer
import faiss

# LangChain imports
from langchain.llms.base import LLM
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
# from langchain import OpenAI  # only for types if you want; not used here

from utils.rag import bioportal_search, bioportal_annotate, candidate_to_text, call_qwen_chat, make_prompt, RetrieverIndex


from dotenv import load_dotenv
load_dotenv()

# -------------------------
# Environment / config
# -------------------------
BIOPORTAL_BASE = os.environ.get("BIOPORTAL_BASE")
BIOPORTAL_KEY = os.environ.get("BIOPORTAL_API_KEY")
QWEN_SERVER_URL = os.environ.get("QWEN_SERVER_URL")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")

# add near top of file with other imports
import re
# from langchain.output_parsers import BaseOutputParser
# from langchain_core.output_parsers import BaseOutputParser
from langchain.schema import AgentFinish

class JSONAgentFinishParserDuck:
    def parse(self, text: str):
        m = re.search(r'\[.*\]', text, flags=re.DOTALL) or re.search(r'\{.*\}', text, flags=re.DOTALL)
        if not m:
            raise ValueError("no JSON found")
        parsed = json.loads(m.group(0))
        return AgentFinish(return_values={"output": parsed}, log="JSONAgentFinishParserDuck")

    def get_format_instructions(self) -> str:
        return "Return only a JSON array or object with the mapping results."



# -------------------------
# Index store (to let tools reference indexes by key)
# -------------------------
INDEX_STORE: Dict[str, RetrieverIndex] = {}

# -------------------------
# LangChain LLM wrapper for Qwen
# -------------------------
class QwenLLM(LLM):
    """LangChain LLM that delegates to call_qwen_chat."""
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        # call Qwen with deterministic settings (temperature=0.0)
        return call_qwen_chat(prompt, server_url=QWEN_SERVER_URL, max_new_tokens=800, temperature=0.0)

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"server_url": QWEN_SERVER_URL}

    @property
    def _llm_type(self) -> str:
        return "qwen"

# -------------------------
# Tool function implementations (these will be wrapped with Tool.from_function)
# -------------------------
def search_bioportal_fn(query: str, ontologies: Optional[str] = None, pagesize: int = 50) -> Dict[str, Any]:
    hits = bioportal_search(query, ontologies=ontologies, pagesize=pagesize)
    return {"ok": True, "count": len(hits), "candidates": hits}

def annotate_bioportal_fn(text: str, ontologies: Optional[str] = None, pagesize: int = 200) -> Dict[str, Any]:
    hits = bioportal_annotate(text, ontologies=ontologies, pagesize=pagesize)
    return {"ok": True, "count": len(hits), "candidates": hits}

def index_build_fn(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    texts = [candidate_to_text(c) for c in candidates]
    retr = RetrieverIndex()
    retr.add(texts, candidates)
    key = f"idx_{int(time.time()*1000)}"
    INDEX_STORE[key] = retr
    return {"ok": True, "index_key": key, "count": len(candidates)}

def index_query_fn(index_key: str, q_text: str, k: int = 8) -> Dict[str, Any]:
    retr = INDEX_STORE.get(index_key)
    if retr is None:
        return {"ok": False, "error": "index_not_found"}
    top = retr.query(q_text, k=k)
    serial = [{"meta": m, "score": float(s)} for m, s in top]
    return {"ok": True, "top_k": serial}

def match_with_qwen_fn(source_label: str, source_def: str, top_k_candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    candidates = [(c["meta"], c["score"]) for c in top_k_candidates]
    prompt = make_prompt(source_label, source_def or "", candidates)
    raw = call_qwen_chat(prompt, max_new_tokens=800, temperature=0.0)
    try:
        parsed = json.loads(raw)
    except Exception:
        try:
            s = raw.index('['); e = raw.rindex(']') + 1
            parsed = json.loads(raw[s:e])
        except Exception as ex:
            return {"ok": False, "error": "failed_parse", "raw": raw}
    # normalize parsed items
    for item in parsed:
        item.setdefault("candidate_uri", None)
        item.setdefault("candidate_label", None)
        item.setdefault("mapping_type", None)
        item.setdefault("confidence", None)
        item.setdefault("reason", None)
    return {"ok": True, "matches": parsed}

def validate_fn(a_label: str, b_label: str, context: str = "") -> Dict[str, Any]:
    prompt = f"Question: Is \"{a_label}\" equivalent to \"{b_label}\"? Context: {context}\nAnswer yes or no and a short reason. Return JSON with keys 'answer' and 'reason'."
    raw = call_qwen_chat(prompt, max_new_tokens=150, temperature=0.0)
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = {"raw": raw}
    return {"ok": True, "validation": parsed}

# -------------------------
# Wrap functions as LangChain Tools
# -------------------------
tool_search = Tool.from_function(func=search_bioportal_fn, name="search_bioportal", description="Search BioPortal for candidate concepts.")
tool_annotate = Tool.from_function(func=annotate_bioportal_fn, name="annotate_bioportal", description="Annotate a free-text span using BioPortal.")
tool_index_build = Tool.from_function(func=index_build_fn, name="index_build", description="Build an in-memory FAISS index from candidate metadata; returns index_key.")
tool_index_query = Tool.from_function(func=index_query_fn, name="index_query", description="Query an existing index by index_key and a query text.")
tool_match = Tool.from_function(func=match_with_qwen_fn, name="match_with_qwen", description="Call Qwen to classify top candidates into mapping types. Returns JSON matches.")
tool_validate = Tool.from_function(func=validate_fn, name="validate", description="Validate a candidate pair with the LLM, returns yes/no and reason.")

TOOLS = [tool_search, tool_annotate, tool_index_build, tool_index_query, tool_match, tool_validate]

# -------------------------
# Initialize agent
# -------------------------
llm = QwenLLM()
agent = initialize_agent(TOOLS, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# attach custom parser so the MRKL planner accepts JSON final outputs
agent.output_parser = JSONAgentFinishParserDuck()

# -------------------------
# Example orchestration function using the agent
# -------------------------
def run_rag_agent_for_label(source_label: str, source_def: str, ontologies: Optional[str] = None, top_k: int = 8):
    pipeline_instruction = f"""
    Goal: Find best mappings for SOURCE: "{source_label}" (definition: "{source_def}").
    Steps the agent should perform (use the provided tools):
      1) call search_bioportal with query equal to the source label (and ontologies={ontologies})
      2) call index_build with the returned candidates
      3) call index_query with the returned index_key and q_text set to "{source_label}\\n{source_def}"
      4) call match_with_qwen with the top_k results from step 3
      5) return the final matches in JSON
    Please only call tools (do not perform external web calls yourself). Produce a final JSON summary of matches.
    """

    result = agent.run(pipeline_instruction)
    return result

# -------------------------
# CLI example
# -------------------------
if __name__ == "__main__":
    src_label = "Femur"
    src_def = "long bone in the leg"
    print("Running agent pipeline for:", src_label)
    out = run_rag_agent_for_label(src_label, src_def, ontologies=None, top_k=8)
    print("Agent output:\n", out)
