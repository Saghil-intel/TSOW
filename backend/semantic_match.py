import os
import re
import hashlib
import threading
import requests

import numpy as np


# ==================================================
# CONFIGURATION
# ==================================================

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest")

EMBEDDING_TIMEOUT = (15, 180)

ROW_MATCH_THRESHOLD = float(os.getenv("ROW_MATCH_THRESHOLD", "0.35"))

TOP_K_ROWS = int(os.getenv("TOP_K_ROWS", "5"))

CACHE_SIZE = int(os.getenv("EMBEDDING_CACHE_SIZE", "10000"))

# nomic-embed-text expects a task prefix on BOTH sides of the comparison.
QUERY_PREFIX = "search_query: "
DOCUMENT_PREFIX = "search_document: "


# ==================================================
# TEXT NORMALIZATION
# ==================================================

def normalize_text(text):

    if not text:
        return ""

    text = re.sub(r"\s+", " ", str(text))

    return text.strip()


def prepare_field_text(field):
    """
    Semantic representation of an identified TSOW field.

    Matching guidance is deliberately NOT included: it contains
    negations ("do not confuse with ...") and embeddings cannot
    interpret negation, so it would pull the vector toward the
    very concepts it warns against. Guidance is given to the LLM
    prompt instead.
    """

    parts = [
        field.get("question", ""),
        field.get("standard_field", ""),
        field.get("description", ""),
    ]

    alternates = field.get("alternate_names", [])
    if alternates:
        parts.append("Also called: " + ", ".join(alternates))

    examples = field.get("examples", [])
    if examples:
        parts.append("Examples: " + ", ".join(examples))

    return normalize_text(
        ". ".join(str(part) for part in parts if part)
    )


# ==================================================
# OLLAMA EMBEDDING CLIENT
# ==================================================

class OllamaEmbeddingClient:

    def __init__(self, base_url=OLLAMA_URL, model=EMBEDDING_MODEL):

        self.base_url = base_url
        self.model = model

        self.session = requests.Session()
        self.lock = threading.RLock()

        print(f"[SEMANTIC] Ollama URL: {self.base_url}")
        print(f"[SEMANTIC] Embedding model: {self.model}")

    def embed(self, text):

        text = normalize_text(text)

        if not text:
            raise ValueError("Cannot generate embedding for empty text.")

        payload = {"model": self.model, "input": text}

        try:

            response = self.session.post(
                f"{self.base_url}/api/embed",
                json=payload,
                timeout=EMBEDDING_TIMEOUT
            )

            response.raise_for_status()

            embeddings = response.json().get("embeddings", [])

            if not embeddings or not embeddings[0]:
                raise ValueError("Ollama returned an empty embedding.")

            vector = np.asarray(embeddings[0], dtype=np.float32)

            if not np.all(np.isfinite(vector)):
                raise ValueError("Embedding contains invalid numeric values.")

            return vector

        except requests.RequestException as e:
            raise RuntimeError(
                f"Ollama embedding request failed: {e}"
            ) from e

    def embed_batch(self, texts):

        texts = [
            normalize_text(text)
            for text in texts
            if normalize_text(text)
        ]

        if not texts:
            return []

        payload = {"model": self.model, "input": texts}

        try:

            response = self.session.post(
                f"{self.base_url}/api/embed",
                json=payload,
                timeout=EMBEDDING_TIMEOUT
            )

            response.raise_for_status()

            embeddings = response.json().get("embeddings", [])

            if len(embeddings) != len(texts):
                raise ValueError("Ollama embedding count mismatch.")

            vectors = []

            for embedding in embeddings:

                vector = np.asarray(embedding, dtype=np.float32)

                if not np.all(np.isfinite(vector)):
                    raise ValueError("Invalid embedding values.")

                norm = np.linalg.norm(vector)

                if norm == 0:
                    raise ValueError("Zero-length embedding.")

                vectors.append(vector / norm)

            return vectors

        except requests.RequestException as e:
            raise RuntimeError(
                f"Ollama batch embedding failed: {e}"
            ) from e

    def close(self):
        self.session.close()


# ==================================================
# SEMANTIC MATCHING ENGINE
# ==================================================

class SemanticMatcher:

    def __init__(self):

        self.client = OllamaEmbeddingClient()
        self.embedding_cache = {}
        self.cache_lock = threading.RLock()

        self._check_model()

    def _check_model(self):

        try:

            response = self.client.session.get(
                f"{OLLAMA_URL}/api/tags",
                timeout=10
            )

            response.raise_for_status()

            models = response.json().get("models", [])

            available = [model.get("name", "") for model in models]

            model_available = any(
                name == self.client.model
                or name.split(":")[0] == self.client.model.split(":")[0]
                for name in available
            )

            if not model_available:
                raise RuntimeError(
                    f"Embedding model '{self.client.model}' "
                    f"not found in Ollama. "
                    f"Available models: {available}"
                )

            print(f"[SEMANTIC] Model verified: {self.client.model}")

        except requests.RequestException as e:
            raise RuntimeError(
                f"Cannot connect to Ollama at {OLLAMA_URL}. "
                f"Make sure Ollama is running. Error: {e}"
            ) from e

    def _cache_key(self, text):

        return hashlib.sha256(
            (self.client.model + "|" + text).encode("utf-8")
        ).hexdigest()

    def _cache_put(self, key, vector):
        # Caller must hold cache_lock.

        if len(self.embedding_cache) >= CACHE_SIZE:
            oldest_key = next(iter(self.embedding_cache))
            self.embedding_cache.pop(oldest_key)

        self.embedding_cache[key] = vector

    def get_embedding(self, text):

        text = normalize_text(text)

        if not text:
            return None

        key = self._cache_key(text)

        with self.cache_lock:
            cached = self.embedding_cache.get(key)
            if cached is not None:
                return cached

        embedding = self.client.embed(text)

        norm = np.linalg.norm(embedding)

        if norm == 0:
            raise ValueError("Ollama returned a zero-length embedding.")

        embedding = embedding / norm

        with self.cache_lock:
            self._cache_put(key, embedding)

        return embedding

    def get_embeddings_batch(self, texts):
        """
        Cache-aware batch embedding. Output order matches input order.
        """

        normalized = [normalize_text(text) for text in texts]

        keys = [
            self._cache_key(text) if text else None
            for text in normalized
        ]

        results = [None] * len(normalized)

        missing_texts = []
        missing_indices = []

        with self.cache_lock:

            for index, key in enumerate(keys):

                if key is None:
                    continue

                cached = self.embedding_cache.get(key)

                if cached is not None:
                    results[index] = cached
                else:
                    missing_texts.append(normalized[index])
                    missing_indices.append(index)

        if missing_texts:

            new_vectors = self.client.embed_batch(missing_texts)

            with self.cache_lock:

                for index, vector in zip(missing_indices, new_vectors):
                    self._cache_put(keys[index], vector)
                    results[index] = vector

        return results

    def similarity(self, text1, text2):

        if not text1 or not text2:
            return 0.0

        a = normalize_text(text1).casefold()
        b = normalize_text(text2).casefold()

        if a == b:
            return 1.0

        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)

        if emb1 is None or emb2 is None:
            return 0.0

        return max(-1.0, min(1.0, float(np.dot(emb1, emb2))))

    def field_embedding(self, field):

        return self.get_embedding(
            QUERY_PREFIX + prepare_field_text(field)
        )

    def retrieve_rows(self, field, rows, top_k=TOP_K_ROWS):
        """
        Returns [(similarity_score, original_row), ...]
        """

        if not rows:
            return []

        field_vector = self.field_embedding(field)

        valid_rows = [
            normalize_text(row)
            for row in rows
            if normalize_text(row)
        ]

        if not valid_rows:
            return []

        # Rows are documents: they need the document prefix so they
        # live in the same embedding space as the query prefix.
        row_vectors = self.get_embeddings_batch(
            [DOCUMENT_PREFIX + row for row in valid_rows]
        )

        scored_rows = [
            (float(np.dot(field_vector, vector)), row)
            for row, vector in zip(valid_rows, row_vectors)
        ]

        scored_rows.sort(key=lambda x: x[0], reverse=True)

        matched_rows = [
            (score, row)
            for score, row in scored_rows
            if score >= ROW_MATCH_THRESHOLD
        ]

        return matched_rows[:top_k]

    def values_agree(self, value1, value2, threshold=0.88):
        """
        Semantic similarity does not guarantee technical equivalence.
        Do not use this alone to compare versions, numbers or specs.
        """

        if not value1 or not value2:
            return False

        a = normalize_text(value1).casefold()
        b = normalize_text(value2).casefold()

        if a == b:
            return True

        return self.similarity(value1, value2) >= threshold

    def warm_up(self, texts):

        self.get_embeddings_batch(texts)

        print(
            f"[SEMANTIC] Warm-up completed. "
            f"Cached embeddings: {len(self.embedding_cache)}"
        )

    def close(self):
        self.client.close()


# ==================================================
# SINGLETON INSTANCE
# ==================================================

_matcher = None
_matcher_lock = threading.Lock()


def get_matcher():

    global _matcher

    if _matcher is None:
        with _matcher_lock:
            if _matcher is None:
                _matcher = SemanticMatcher()

    return _matcher


# ==================================================
# PUBLIC FUNCTIONS
# ==================================================

def semantic_retrieve_rows(field, rows, top_k=TOP_K_ROWS):

    return get_matcher().retrieve_rows(field, rows, top_k)


def semantic_match_rows(field, rows, top_k=TOP_K_ROWS):
    """
    Public interface used by llm.py.
    Returns the list of relevant questionnaire rows.
    """

    matched_rows = get_matcher().retrieve_rows(
        field=field,
        rows=rows,
        top_k=top_k
    )

    print(f"[SEMANTIC] Field: {field.get('question')}")

    for score, row in matched_rows:
        print(f"    Similarity: {score:.4f} | {row[:150]}")

    return [row for score, row in matched_rows]