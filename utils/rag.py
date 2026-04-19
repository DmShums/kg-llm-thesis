"""
RAG-based ontology matching utilities.
Some code parts are actively utilised in notebooks like `run_repairer.ipynb` and `run_validator.ipynb`,
while others, like bioportal_search or bioportal_annotate, are more experimental or for future use.
"""

import os
import time
import requests
import json
from typing import List, Dict, Any, Tuple

# Embedding + vector store
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

from dotenv import load_dotenv
load_dotenv()

BIOPORTAL_BASE = os.environ.get("BIOPORTAL_BASE")
BIOPORTAL_KEY = os.environ.get("BIOPORTAL_API_KEY")
QWEN_SERVER_URL = os.environ.get("QWEN_SERVER_URL")
EMBED_MODEL = os.environ.get("EMBED_MODEL")

import re

def _extract_json_text(text: str) -> str:
    # prefer array if present
    m = re.search(r'\[.*\]', text, flags=re.DOTALL)
    if m:
        return m.group(0)
    m = re.search(r'\{.*\}', text, flags=re.DOTALL)
    if m:
        return m.group(0)
    return text

# -------------------------
# 1) Search / Annotate BioPortal API
# -------------------------

def _bioportal_headers() -> Dict[str, str]:
    if not BIOPORTAL_KEY:
        raise RuntimeError("BIOPORTAL_API_KEY is not set")
    return {"Authorization": f"apikey token={BIOPORTAL_KEY}"}


def bioportal_search(query: str, ontologies: str = None, pagesize: int = 50) -> List[Dict[str, Any]]:
    headers = _bioportal_headers()
    params = {"q": query, "pagesize": pagesize}
    if ontologies:
        params["ontologies"] = ontologies
    resp = requests.get(f"{BIOPORTAL_BASE}/search", headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    items = data.get("collection") or data.get("@graph") or data.get("results") or data.get("search")
    if items is None:
        return [data]
    return items

def bioportal_annotate(text: str,
                       ontologies: str = None,
                       semantic_types: str = None,
                       longest_only: bool = True,
                       expand_mappings: bool = False,
                       minimum_match_length: int = 0,
                       pagesize: int = 200) -> List[Dict[str, Any]]:
    """
    Calls the BioPortal Annotator endpoint and returns a list of annotation objects.
    """
    headers = _bioportal_headers()
    params = {
        "text": text,
        "longest_only": "true" if longest_only else "false",
        "expand_mappings": "true" if expand_mappings else "false",
        "minimum_match_length": str(minimum_match_length),
        "pagesize": str(pagesize)
    }
    if ontologies:
        params["ontologies"] = ontologies
    if semantic_types:
        params["semantic_types"] = semantic_types

    resp = requests.get(f"{BIOPORTAL_BASE}/annotator", headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    items = data.get("annotations") or data.get("collection") or data.get("matches") or data.get("results")
    if items is None:
        return [data]
    return items


# -------------------------
# 1b) BioPortal Recommender / Ontology metadata (for Mediating Ontology selection)
# -------------------------

def bioportal_recommender(
    text: str,
    input_ontologies: str  = None,
    output_ontologies: str  = None,
    max_elements: int = 15,
) -> List[Dict[str, Any]]:
    """
    Calls the BioPortal Recommender endpoint.

    This is useful for selecting a mediating ontology (MO) given some text derived from O1/O2.

    Notes:
    - BioPortal's API returns a ranked list of ontologies with scores.
    - Parameters can vary by BioPortal deployment; we keep them optional and tolerant.
    """
    headers = _bioportal_headers()
    params: Dict[str, Any] = {
        "input": text,
        "maxElements": str(max_elements),
    }
    if input_ontologies:
        params["input_ontologies"] = input_ontologies
    if output_ontologies:
        params["output_ontologies"] = output_ontologies

    resp = requests.get(f"{BIOPORTAL_BASE}/recommender", headers=headers, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    # Common shapes: list[dict] or {"collection": [...]}
    if isinstance(data, dict):
        items = data.get("collection") or data.get("@graph") or data.get("results")
        return items if isinstance(items, list) else [data]
    return data if isinstance(data, list) else [data]


def bioportal_get_ontology(acronym: str) -> Dict[str, Any]:
    """Fetch BioPortal ontology metadata for an acronym (e.g., 'FMA', 'SNOMEDCT')."""
    headers = _bioportal_headers()
    resp = requests.get(f"{BIOPORTAL_BASE}/ontologies/{acronym}", headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


def bioportal_get_latest_submission(acronym: str) -> Dict[str, Any]:
    """Fetch latest submission metadata, which typically contains download links."""
    headers = _bioportal_headers()
    resp = requests.get(f"{BIOPORTAL_BASE}/ontologies/{acronym}/latest_submission", headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


def bioportal_download_latest_submission(
    acronym: str,
    out_path: str,
    chunk_size: int = 1024 * 1024,
    fallback_ontology_json: dict = None,
) -> str:
    """
    Download ontology from BioPortal.

    Tries (in order):
    1) latest_submission.links.download
    2) fallback_ontology_json.links.download
    3) ontology-level /ontologies/{ACRONYM}/download   ← REQUIRED for NCIT
    """

    headers = _bioportal_headers()
    download_url = None

    # 1. latest submission download
    try:
        sub = bioportal_get_latest_submission(acronym)
        download_url = sub.get("links", {}).get("download")
    except Exception:
        pass

    # 2. fallback ontology JSON (e.g. recommender)
    if not download_url and fallback_ontology_json:
        download_url = fallback_ontology_json.get("links", {}).get("download")

    # 3. ontology-level download (NCIT, MeSH, etc.)
    if not download_url:
        download_url = f"https://data.bioontology.org/ontologies/{acronym}/download"

    # --- download ---
    with requests.get(download_url, headers=headers, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)

    return out_path

# -------------------------
# 2) Normalize candidate -> text context
# -------------------------
def candidate_to_text(c: Dict[str, Any]) -> str:
    parts = []
    pref = c.get("prefLabel") or c.get("label") or c.get("name")
    if pref:
        parts.append(f"Label: {pref}")
    syns = c.get("synonym") or c.get("synonyms") or []
    if isinstance(syns, list) and syns:
        parts.append("Synonyms: " + "; ".join(map(str, syns[:10])))
    defs = c.get("definition") or c.get("definitions")
    if isinstance(defs, list) and defs:
        parts.append("Definition: " + (defs[0] if isinstance(defs[0], str) else json.dumps(defs[0])))
    elif isinstance(defs, str):
        parts.append("Definition: " + defs)
    ont = c.get("links", {}).get("ontology") or c.get("ontology")
    if ont:
        parts.append(f"Ontology: {ont}")
    uri = c.get("@id") or c.get("id") or c.get("iri") or c.get("resource")
    if uri:
        parts.append(f"URI: {uri}")
    mappings = c.get("mappings") or c.get("mappingsFrom") or []
    if mappings:
        parts.append("Mappings: " + json.dumps(mappings)[:200])
    return "\n".join(parts)

# -------------------------
# 3) Build embeddings + FAISS index
# -------------------------
class RetrieverIndex:
    def __init__(self, embed_model_name: str = EMBED_MODEL):
        self.encoder = SentenceTransformer(embed_model_name)
        self.dim = self.encoder.get_sentence_embedding_dimension()
        self.index = None
        self.metadata: List[Dict[str, Any]] = []

    def add(self, texts: List[str], metas: List[Dict[str, Any]]):
        vs = self.encoder.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        faiss.normalize_L2(vs)
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(vs)
        self.metadata.extend(metas)

    def query(self, q_text: str, k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        v = self.encoder.encode([q_text], convert_to_numpy=True)
        faiss.normalize_L2(v)
        if self.index is None:
            return []
        D, I = self.index.search(v, k)
        results = []
        for idx, score in zip(I[0], D[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            results.append((self.metadata[idx], float(score)))
        return results
    
    def inspect(self, n: int = 5):
        """Prints up to `n` entries stored in the FAISS index."""
        if not self.metadata:
            print("Vector DB is empty.")
            return
        print(f"Stored {len(self.metadata)} entries (dim={self.dim}) in vector DB:")
        for i, meta in enumerate(self.metadata[:n]):
            label = meta.get("prefLabel") or meta.get("label")
            uri = meta.get("@id") or meta.get("iri")
            print(f"{i+1:>3}. {label}  ({uri})")


# -------------------------
# 4) Qwen chat call (replaces OpenAI call)
# -------------------------
def call_qwen_chat(prompt: str,
                   server_url: str = QWEN_SERVER_URL,
                   max_new_tokens: int = 800,
                   temperature: float = 0.0,
                   timeout: int = 300) -> str:
    system_msg = {
        "role": "system",
        "content": (
            "You are an expert biomedical ontology matcher. Output JSON only: a JSON array of objects "
            "with fields {candidate_uri, candidate_label, mapping_type, confidence, reason}."
        )
    }
    user_msg = {"role": "user", "content": prompt}
    payload = {
        "messages": [system_msg, user_msg],
        "max_new_tokens": max_new_tokens,
        "temperature": float(temperature)
    }
    resp = requests.post(server_url, json=payload, timeout=timeout)
    resp.raise_for_status()

    # Normalize server response
    try:
        j = resp.json()
        if isinstance(j, dict) and "response" in j and isinstance(j["response"], str):
            text = j["response"]
        else:
            text = json.dumps(j, ensure_ascii=False)
    except ValueError:
        text = resp.text

    # Extract JSON block (array or object)
    extracted = _extract_json_text(text).strip()

    # Try to parse and normalize to a JSON array string
    try:
        parsed = json.loads(extracted)
        if isinstance(parsed, dict):
            # Wrap single object into a list
            return json.dumps([parsed], ensure_ascii=False)
        elif isinstance(parsed, list):
            return json.dumps(parsed, ensure_ascii=False)
        else:
            # unexpected scalar -> fallback to original extracted
            return extracted
    except Exception:
        # If parsing fails, return the raw extracted text so downstream code can attempt its heuristics
        return extracted

def chat_invoke(messages: list[dict], server_url: str = QWEN_SERVER_URL, max_new_tokens: int = 50, temperature: float = 0.7) -> str:
    payload = {
        "messages": messages,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature
    }
    response = requests.post(f"{server_url}", json=payload, timeout=300)
    response.raise_for_status()
    data = response.json()
    return data["response"]


# -------------------------
# 5) Prompt construction & pipeline
# -------------------------
def make_prompt(source_label: str, source_def: str, candidates: List[Tuple[Dict[str, Any], float]]) -> str:
    header = (
        "For each candidate below, decide whether it is an EXACT MATCH, BROADER, NARROWER, RELATED, or NO_MATCH "
        "to the SOURCE concept. Return a JSON array of objects: "
        "[{candidate_uri, candidate_label, mapping_type, confidence (0-1), reason}].\n\n"
    )
    src = f"SOURCE:\nLabel: {source_label}\nDefinition: {source_def}\n\n"
    cand_texts = []
    for i, (c, score) in enumerate(candidates, start=1):
        text = candidate_to_text(c)
        cand_texts.append(f"Candidate {i} (score={score:.3f}):\n{text}\n")
    instructions = (
        "\nImportant: be concise in 'reason'. Use mapping_type exactly from {EXACT, BROADER, NARROWER, RELATED, NO_MATCH}.\n"
        "Return JSON only (no extra commentary).\n"
    )
    return header + src + "\n".join(cand_texts) + instructions


# -------------------------
# 6) RAG match pipeline
# -------------------------

def rag_match_for_term_search(source_label: str, source_def: str, ontologies: str = None, top_k: int = 8) -> List[Dict[str, Any]]:
    # 1. Search BioPortal for text candidates
    raw_candidates = bioportal_search(source_label, ontologies=ontologies, pagesize=top_k * 3)
    candidates_text = [candidate_to_text(c) for c in raw_candidates]

    # 2. Build in-memory index from those candidates
    retr = RetrieverIndex()
    retr.add(candidates_text, raw_candidates)

    # 3. Retrieve top-k by embedding similarity
    q_text = f"{source_label}\n{source_def or ''}"
    top = retr.query(q_text, k=top_k)

    # 4. Build prompt and call Qwen server
    prompt = make_prompt(source_label, source_def or "", top)
    qwen_output = call_qwen_chat(prompt, max_new_tokens=800, temperature=0.0)

    # 5. Parse JSON from model output (robust: accept array or single object)
    try:
        parsed = json.loads(qwen_output)
    except Exception:
        # try extract JSON substring (prefer array then object)
        try:
            start = qwen_output.index('[')
            end = qwen_output.rindex(']') + 1
            parsed = json.loads(qwen_output[start:end])
        except Exception:
            try:
                start = qwen_output.index('{')
                end = qwen_output.rindex('}') + 1
                parsed = json.loads(qwen_output[start:end])
            except Exception as e:
                raise RuntimeError("Failed to parse model JSON output:\n" + qwen_output) from e

    # Normalize to a list
    if isinstance(parsed, dict):
        parsed = [parsed]
    elif not isinstance(parsed, list):
        raise RuntimeError("Parsed model output is neither a list nor an object:\n" + str(parsed))

    # 5b. Validate / canonicalize parsed items
    expected_keys = {"candidate_uri", "candidate_label", "mapping_type", "confidence", "reason"}
    for item in parsed:
        if not isinstance(item, dict):
            raise RuntimeError(f"Parsed item is not an object: {item}")
        # fill missing keys with None
        missing = expected_keys - set(item.keys())
        for k in missing:
            item[k] = None
        # coerce confidence to float in [0,1]
        try:
            conf = float(item.get("confidence") if item.get("confidence") is not None else 0.0)
            # clamp
            conf = max(0.0, min(1.0, conf))
        except Exception:
            conf = 0.0
        item["confidence"] = conf
        # ensure labels are strings
        item["candidate_label"] = (item.get("candidate_label") or "").strip()
        item["mapping_type"] = (item.get("mapping_type") or "").strip()
        item["reason"] = (item.get("reason") or "").strip()

    # 6. Optionally enrich parsed results with candidate info (URI) if missing
    # build map by label and uri as fallback
    id_map = {}
    for c in raw_candidates:
        key = (c.get("prefLabel") or c.get("label") or "").strip()
        if key:
            id_map[key] = c

    enriched = []
    for item in parsed:
        label = item.get("candidate_label", "").strip()
        meta = id_map.get(label)
        if meta and not item.get("candidate_uri"):
            item["candidate_uri"] = meta.get("@id") or meta.get("id") or meta.get("iri")
        enriched.append(item)
    return enriched


def rag_match_for_term_annotator(source_text: str,
                                 ontologies: str = None,
                                 top_k: int = 8) -> List[Dict[str, Any]]:
    # 1. Annotate text using BioPortal Annotator
    raw_candidates = bioportal_annotate(source_text, ontologies=ontologies, pagesize=top_k * 3)
    
    # 2. Convert annotations to text for embedding
    candidates_text = [candidate_to_text(c) for c in raw_candidates]

    # 3. Build in-memory FAISS index
    retr = RetrieverIndex()
    retr.add(candidates_text, raw_candidates)

    # 4. Retrieve top-k by embedding similarity
    top = retr.query(source_text, k=top_k)

    # 5. Build prompt and call Qwen server
    prompt = make_prompt(source_text, "", top)
    qwen_output = call_qwen_chat(prompt, max_new_tokens=800, temperature=0.0)

    # 6. Parse JSON output (robust)
    try:
        parsed = json.loads(qwen_output)
    except Exception:
        try:
            start = qwen_output.index('[')
            end = qwen_output.rindex(']') + 1
            parsed = json.loads(qwen_output[start:end])
        except Exception:
            try:
                start = qwen_output.index('{')
                end = qwen_output.rindex('}') + 1
                parsed = json.loads(qwen_output[start:end])
            except Exception as e:
                raise RuntimeError("Failed to parse model JSON output:\n" + qwen_output) from e

    # Normalize to a list
    if isinstance(parsed, dict):
        parsed = [parsed]
    elif not isinstance(parsed, list):
        raise RuntimeError("Parsed model output is neither a list nor an object:\n" + str(parsed))

    # Validate / canonicalize parsed items
    expected_keys = {"candidate_uri", "candidate_label", "mapping_type", "confidence", "reason"}
    for item in parsed:
        if not isinstance(item, dict):
            raise RuntimeError(f"Parsed item is not an object: {item}")
        missing = expected_keys - set(item.keys())
        for k in missing:
            item[k] = None
        try:
            conf = float(item.get("confidence") if item.get("confidence") is not None else 0.0)
            conf = max(0.0, min(1.0, conf))
        except Exception:
            conf = 0.0
        item["confidence"] = conf
        item["candidate_label"] = (item.get("candidate_label") or "").strip()
        item["mapping_type"] = (item.get("mapping_type") or "").strip()
        item["reason"] = (item.get("reason") or "").strip()

    # 7. Enrich with official URIs from raw_candidates
    label_to_uri = { (c.get("prefLabel") or c.get("label") or "").strip(): (c.get("@id") or c.get("id") or c.get("iri"))
                    for c in raw_candidates }
    enriched = []
    for item in parsed:
        label = item.get("candidate_label", "").strip()
        if label in label_to_uri and label_to_uri[label]:
            item["candidate_uri"] = label_to_uri[label]
        else:
            # keep whatever the model returned or None
            item["candidate_uri"] = item.get("candidate_uri")
        enriched.append(item)
    
    return enriched



# few-shot rag utilities

import json
from typing import List, Dict
from itertools import product

from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# -------------------------
# Preprocess few-shot examples
# -------------------------
from utils.onto_object import OntologyEntryAttr, OntologyAccess

def extract_direct_entity(example):
    text = example["input"]

    src_label = re.search(r'first one is "(.*?)"', text).group(1)
    src_parent = re.search(r'broader category "(.*?)"', text).group(1)

    tgt_label = re.search(r'second one is "(.*?)"', text).group(1)
    tgt_parent = re.findall(r'broader category "(.*?)"', text)[1]

    return src_label, tgt_label, src_parent, tgt_parent


# -------------------------
# Build vector store
# -------------------------

def build_vectorstore(few_shot_samples: List[Dict]) -> FAISS:
    """
    Build FAISS vector store from few-shot examples.
    """

    if not few_shot_samples:
        raise ValueError("few_shot_samples is empty")

    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/e5-base-v2",
        encode_kwargs={"normalize_embeddings": True}
    )

    docs: List[Document] = []

    for i, sample in enumerate(few_shot_samples):

        if "input" not in sample:
            raise ValueError("Few-shot sample missing 'input' field")
        
        src, tgt, src_p, tgt_p = extract_direct_entity(sample)

        def normalize(x: str) -> str:
            if not x:
                return ""
            return (
                x.replace("_", " ")
                .replace("-", " ")
                .lower()
                .strip()
            )
        
        src = normalize(src)
        tgt = normalize(tgt)
        src_p = normalize(src_p)
        tgt_p = normalize(tgt_p)

        # basic content structure
        # content = (
        #     f"source entity {src} parent {src_p} "
        #     f"target entity {tgt} parent {tgt_p}"
        # )

        content = f"""
passage: Possible ontology alignment:

Source entity: {src}
Parent class: {src_p}

Target entity: {tgt}
Parent class: {tgt_p}

Are these biomedical concepts equivalent?
"""

        docs.append(
            Document(
                page_content=content,
                metadata={
                    "index": i,
                    "src": src,
                    "tgt": tgt,
                    "label": sample["output"],
                    "sample": sample
                },
            )
        )

    vectorstore = FAISS.from_documents(docs, embeddings)

    return vectorstore


# -------------------------
# Expand experiment setups
# -------------------------

def expand_setups(
    setups: List[Dict],
    base_setup: Dict,
    exclude_keys: List[str] = None
) -> List[Dict]:

    base_setup = base_setup or {}
    exclude_keys = exclude_keys or []

    expanded: List[Dict] = []

    for setup in setups:

        constant_part: Dict = {}
        keys: List[str] = []
        values: List[List] = []

        setup = {**base_setup, **setup}

        for k, v in setup.items():

            if isinstance(v, list) and k not in exclude_keys:
                keys.append(k)
                values.append(v)
            else:
                constant_part[k] = v

        if not keys:
            expanded.append(constant_part)
            continue

        for combo in product(*values):

            new_setup = constant_part.copy()
            new_setup.update(dict(zip(keys, combo)))

            expanded.append(new_setup)

    return expanded


# -------------------------
# Load few-shot examples
# -------------------------

def load_few_shot_examples(path: str, prompt_key: str) -> List[Dict]:
    """
    Load few-shot examples from dataset file.
    """

    with open(path, "r") as f:
        data = json.load(f)

    examples = data.get("OpenAI", {}).get(prompt_key, [])

    if not isinstance(examples, list):
        raise ValueError(f"{prompt_key} must contain a list of examples")

    return examples
