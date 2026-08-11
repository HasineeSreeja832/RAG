"""
Central configuration for the RAG application.

Keeping every tunable value in one place makes the system easy to reason
about and easy to defend in a viva: "why chunk_size=1000?" has one answer,
not five scattered across the codebase.
"""

from pathlib import Path

# --- Storage locations ---
BASE_DIR = Path(__file__).resolve().parent.parent
PERSIST_DIR = str(BASE_DIR / "chroma_db")       # persistent vector store
UPLOAD_DIR = BASE_DIR / "temp_uploads"           # scratch space for uploaded PDFs
UPLOAD_DIR.mkdir(exist_ok=True)

# --- Ollama connection ---
OLLAMA_BASE_URL = "http://localhost:11434"

# --- Model choices exposed in the UI ---
EMBEDDING_MODEL = "nomic-embed-text"
AVAILABLE_LLM_MODELS = ["llama3.2", "llama3.1", "mistral", "phi3"]
DEFAULT_LLM_MODEL = "llama3.2"

# --- Chunking defaults (tunable from the sidebar) ---
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200

# --- Retrieval defaults ---
DEFAULT_TOP_K = 4
DEFAULT_TEMPERATURE = 0.0

# Hybrid retrieval: combine keyword search (BM25) with vector similarity.
# Vector embeddings are great at "meaning" but often miss exact keywords,
# numbers, codes, or names (e.g. "30 days", "SKU-1042") because those don't
# carry much semantic weight. BM25 is the opposite: great at exact terms,
# blind to paraphrasing. Blending both catches more of what a pure-vector
# retriever misses, which is a common cause of "vague/wrong" RAG answers.
USE_HYBRID_SEARCH = True
BM25_WEIGHT = 0.4
VECTOR_WEIGHT = 0.6

# MMR (Maximal Marginal Relevance) trades a little pure-similarity for
# diversity, so the top-k isn't 4 near-duplicate chunks of the same
# paragraph. fetch_k is how many candidates MMR considers before picking
# the final k.
MMR_FETCH_K_MULTIPLIER = 4

# When reranking is on, retrieve this many times top_k as candidates before
# the cross-encoder narrows them back down to top_k. Wider net = better
# chance the right chunk is in the pool for the reranker to find.
RERANK_CANDIDATE_MULTIPLIER = 4

# --- Chroma collection name ---
# A single collection is used so that documents persist and accumulate
# across sessions until the user explicitly clears the knowledge base.
COLLECTION_NAME = "rag_knowledge_base"
