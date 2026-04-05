# Advancing Knowledge Graph Alignment Using Large Language Models

Research codebase for **ontology matching** experiments: run **LogMap** / **LogMapBIO**, validate candidate mappings with an **LLM**, rank LogMap logical repair plans with an LLM, suggest mediating ontologies via BioPortal, and evaluate against OAEI-style reference alignments (`full.tsv`).

Primary workflow: **Jupyter notebooks** at the repo root, plus importable **Python modules** under `modules/` and `utils/`.

---

## Capabilities

| Capability | Where |
|------------|-------|
| LogMap **MATCHER** (Docker Corretto 8) | `modules/logmap_wrapper.py` → `run_logmap_alignment` |
| LogMap **local** (Java 8; default macOS Temurin path) | `run_logmap_alignment_locally` |
| LogMapBIO **MATCHER-BIO** | `run_logmap_bio` → JAR under `modules/logmap-bio/`, `JAVA_EXE` in `.env` |
| LLM **yes/no** validation of pairs (multiple prompt styles) | `modules/llm_validator.py` + `utils/prompts/` |
| LLM **repair-plan ranking** | `modules/logical_repairer.py` + `utils/prompts/logical_repair_prompt.py` |
| Mediating ontology **search / ranking** (BioPortal) | `modules/mediating_selector.py` + `utils/rag.py` |
| Metrics, confusion matrices, CSV exports | `modules/evaluator.py` |
| Ontology loading / labels (OWL) | `utils/onto_access.py`, `utils/onto_object.py` |

### LLM Backends

| Backend | Module | Key |
|---------|--------|-----|
| **OpenRouter** (OpenAI-compatible) | `utils/llm_server/open_router.py` | `OPENROUTER_API_KEY` |
| **Local Qwen HTTP API** | `utils/llm_server/qwen.py` | `QWEN_SERVER_URL` |
| **Google Gemini** (OpenAI-compatible) | `utils/llm_server/gemini.py` | `GEMINI_API_KEY` |
| Local server helper | `utils/llm_server/qwen_server_runner.py` | — |

---

## Repository Layout

| Path | Role |
|------|------|
| `modules/` | Pipelines: `logmap_wrapper.py`, `logical_repairer.py`, `llm_validator.py`, `evaluator.py`, `mediating_selector.py`, `logmap_bio_compose_runner.py` |
| `modules/logmap/` | Place **`logmap-matcher-4.0.jar`** here for standard LogMap |
| `modules/logmap-bio/` | Place **`logmap-matcher-4.0.jar`** here for LogMapBIO |
| `utils/` | Ontology access, RAG/BioPortal (`rag.py`), prompts (`utils/prompts/`), LLM clients (`utils/llm_server/`) |
| `data/` | Ontology subsets, reference alignments (`refs_equiv/full.tsv`), LogMap repair dumps (`data/LogMap_repair_plans/`) |
| `output/` | Default run outputs (mappings, mediator cache, repair JSON, etc.) |
| `final_results/` | Curated evaluation bundles (metrics CSVs, `prompts_used.txt`, etc.) |
| `archive/` | Older notebook copies |

### Entry Points

| Notebook | Purpose |
|----------|---------|
| `run_logmap.ipynb` | LogMap / LogMapBIO alignment |
| `run_validator.ipynb` | LLM validation experiments |
| `run_repairer.ipynb` | LLM + heuristic repair-plan selection and evaluation |
| `run_mediator.ipynb` | Mediating ontology selection |
| `run_local_validator.ipynb` | Local validator workflows |

---

## Prerequisites

- **Python 3** with a scientific stack. No `requirements.txt` is checked in; typical dependencies include:
  `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `tqdm`, `python-dotenv`, `pydantic`, `owlready2`, `openai`, `requests`, `httpx`, `jupyter`
  For BioPortal / RAG features: `sentence-transformers`, `faiss-cpu`
- **Java 8** for LogMap JARs
- **Docker** (optional, but default for `run_logmap_alignment`): image `amazoncorretto:8-alpine`, repo mounted at `/workspace`, ~12 GB container memory recommended for `-Xmx10g`

---

## Environment (`.env`)
```env
# Repo root for Docker / consistent relative paths (optional; defaults to project root)
WORKSPACE_ROOT=

# LogMapBIO / local Java
JAVA_EXE=/path/to/java8

# OpenRouter (LLMValidator)
OPENROUTER_API_KEY=

# Local Qwen server (LogicalRepairer default client)
QWEN_SERVER_URL=http://localhost:8000/chat

# Google Gemini client (optional)
GEMINI_API_KEY=

# BioPortal (mediating selector / RAG)
BIOPORTAL_API_KEY=
BIOPORTAL_BASE=https://data.bioontology.org

# Embeddings for RAG (if used)
EMBED_MODEL=
```

---

## Data

Large `.owl` files are not committed. Reference alignments are tab-separated URI pairs (no header), typically at:

```
data/<dataset>/<subset>/refs_equiv/full.tsv
```

LogMap repair plan text dumps used by the repairer live under `data/LogMap_repair_plans/`.

---

## LogMap Setup

### Docker (`run_logmap_alignment`)

Mounts `WORKSPACE_ROOT` to `/workspace` and runs:
```bash
java -Xmx10g -jar modules/logmap/logmap-matcher-4.0.jar \
  MATCHER file:<o1> file:<o2> <output_dir>/ true
```

Expected outputs include `logmap_mappings.rdf` and `logmap_mappings_to_ask_oracle_user_llm.txt`.

### Local Java (`run_logmap_alignment_locally`)

Uses `JAVA8_BIN` hardcoded in `modules/logmap_wrapper.py` (Temurin 8 on macOS). Adjust the path for your machine or use Docker for portability.

### LogMapBIO (`run_logmap_bio`)

Requires `JAVA_EXE` and `modules/logmap-bio/logmap-matcher-4.0.jar`. Mode: `MATCHER-BIO` with final argument `dummy` (see `logmap_wrapper.py`).

---

## Outputs

| Path | Contents |
|------|----------|
| `output/` | Fresh LogMap runs, mediator caches, repair JSON |
| `final_results/` | Structured evaluation folders — `metrics.csv`, `detailed_results.csv`, `prompts_used.txt` |

---

## License / Attribution
