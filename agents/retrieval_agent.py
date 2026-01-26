"""
RetrievalAgent: Handles candidate retrieval from BioPortal and local ontologies.
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from utils.rag import (
    bioportal_search,
    bioportal_annotate,
    candidate_to_text,
    RetrieverIndex
)
from utils.onto_access import OntologyAccess
from utils.onto_object import OntologyEntryAttr
from utils.constants import LOGGER


class RetrievalAgent:
    """
    RetrievalAgent handles candidate retrieval from multiple sources:
    - BioPortal API (search and annotator)
    - Local ontology files (direct access)
    - Vector index (FAISS) for semantic search
    """

    def __init__(
        self,
        use_bioportal: bool = True,
        use_local_ontology: bool = True,
        embed_model_name: Optional[str] = None
    ):
        self.use_bioportal = use_bioportal
        self.use_local_ontology = use_local_ontology
        self.embed_model_name = embed_model_name or os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")
        self._vector_indexes: Dict[str, RetrieverIndex] = {}
        self._local_ontologies: Dict[str, OntologyAccess] = {}

    def load_local_ontology(self, ontology_path: str, ontology_id: str) -> OntologyAccess:
        """Load a local ontology file."""
        if ontology_id in self._local_ontologies:
            return self._local_ontologies[ontology_id]
        
        LOGGER.info(f"Loading ontology {ontology_id} from {ontology_path}")
        onto = OntologyAccess(ontology_path, annotate_on_init=True)
        self._local_ontologies[ontology_id] = onto
        return onto

    def search_bioportal(
        self,
        query: str,
        ontologies: Optional[str] = None,
        pagesize: int = 50
    ) -> List[Dict[str, Any]]:
        """Search BioPortal for candidates."""
        if not self.use_bioportal:
            return []
        
        try:
            candidates = bioportal_search(query, ontologies=ontologies, pagesize=pagesize)
            LOGGER.info(f"BioPortal search returned {len(candidates)} candidates for '{query}'")
            return candidates
        except Exception as e:
            LOGGER.error(f"BioPortal search failed: {e}")
            return []

    def annotate_bioportal(
        self,
        text: str,
        ontologies: Optional[str] = None,
        pagesize: int = 200
    ) -> List[Dict[str, Any]]:
        """Annotate text using BioPortal annotator."""
        if not self.use_bioportal:
            return []
        
        try:
            annotations = bioportal_annotate(text, ontologies=ontologies, pagesize=pagesize)
            LOGGER.info(f"BioPortal annotator returned {len(annotations)} annotations")
            return annotations
        except Exception as e:
            LOGGER.error(f"BioPortal annotation failed: {e}")
            return []

    def search_local_ontology(
        self,
        query: str,
        ontology_id: str,
        max_results: int = 50
    ) -> List[Dict[str, Any]]:
        """Search local ontology by name/URI."""
        if not self.use_local_ontology or ontology_id not in self._local_ontologies:
            return []
        
        onto = self._local_ontologies[ontology_id]
        candidates = []
        
        # Search by name
        classes = onto.getClassObjectsContainingName(query)
        for cls in classes[:max_results]:
            entry = OntologyEntryAttr(class_uri=None, onto_entry=cls, onto=onto)
            candidates.append({
                "@id": cls.iri,
                "prefLabel": next(iter(entry.get_preffered_names()), cls.name),
                "synonym": list(entry.get_synonyms()),
                "definition": list(entry.annotation.get("all_names", set())),
                "ontology": ontology_id,
                "source": "local"
            })
        
        LOGGER.info(f"Local ontology search returned {len(candidates)} candidates")
        return candidates

    def build_vector_index(
        self,
        candidates: List[Dict[str, Any]],
        index_key: Optional[str] = None
    ) -> str:
        """Build a FAISS vector index from candidates."""
        if not index_key:
            import time
            index_key = f"idx_{int(time.time() * 1000)}"
        
        texts = [candidate_to_text(c) for c in candidates]
        retr = RetrieverIndex(embed_model_name=self.embed_model_name)
        retr.add(texts, candidates)
        self._vector_indexes[index_key] = retr
        
        LOGGER.info(f"Built vector index '{index_key}' with {len(candidates)} candidates")
        return index_key

    def query_vector_index(
        self,
        index_key: str,
        query_text: str,
        k: int = 8
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Query a vector index for similar candidates."""
        if index_key not in self._vector_indexes:
            LOGGER.warning(f"Index '{index_key}' not found")
            return []
        
        retr = self._vector_indexes[index_key]
        results = retr.query(query_text, k=k)
        return results

    def retrieve_candidates(
        self,
        source_label: str,
        source_def: Optional[str] = None,
        target_ontology_id: Optional[str] = None,
        use_search: bool = True,
        use_annotator: bool = False,
        top_k: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Main retrieval method that combines multiple sources.
        
        Args:
            source_label: Label of the source concept
            source_def: Optional definition of the source concept
            target_ontology_id: Optional target ontology ID to restrict search
            use_search: Whether to use BioPortal search
            use_annotator: Whether to use BioPortal annotator
            top_k: Maximum number of candidates to return
        
        Returns:
            List of candidate dictionaries
        """
        all_candidates = []
        
        # BioPortal search
        if use_search:
            bioportal_candidates = self.search_bioportal(
                source_label,
                ontologies=target_ontology_id,
                pagesize=top_k * 2
            )
            all_candidates.extend(bioportal_candidates)
        
        # BioPortal annotator
        if use_annotator:
            query_text = f"{source_label}\n{source_def or ''}"
            annotator_candidates = self.annotate_bioportal(
                query_text,
                ontologies=target_ontology_id,
                pagesize=top_k * 2
            )
            all_candidates.extend(annotator_candidates)
        
        # Local ontology search
        if target_ontology_id and self.use_local_ontology:
            local_candidates = self.search_local_ontology(
                source_label,
                target_ontology_id,
                max_results=top_k
            )
            all_candidates.extend(local_candidates)
        
        # Deduplicate by URI
        seen_uris = set()
        unique_candidates = []
        for cand in all_candidates:
            uri = cand.get("@id") or cand.get("id") or cand.get("iri")
            if uri and uri not in seen_uris:
                seen_uris.add(uri)
                unique_candidates.append(cand)
            elif not uri:
                # Keep candidates without URI (might be duplicates but safer)
                unique_candidates.append(cand)
        
        LOGGER.info(f"Retrieved {len(unique_candidates)} unique candidates (from {len(all_candidates)} total)")
        return unique_candidates[:top_k]
