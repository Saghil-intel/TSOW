
import os
import re
import hashlib
import threading
import requests

import numpy as np


# ==================================================
# CONFIGURATION
# ==================================================

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434"
).rstrip("/")

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "nomic-embed-text:latest"
)

EMBEDDING_TIMEOUT = (15, 180)

ROW_MATCH_THRESHOLD = float(
    os.getenv("ROW_MATCH_THRESHOLD", "0.35")
)

TOP_K_ROWS = int(
    os.getenv("TOP_K_ROWS", "5")
)

CACHE_SIZE = int(
    os.getenv("EMBEDDING_CACHE_SIZE", "10000")
)


# ==================================================
# TEXT NORMALIZATION
# ==================================================

def normalize_text(text):

    if not text:
        return ""

    text = str(text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def prepare_field_text(field):
    """
    Construct the semantic representation of an
    already identified TSOW field.

    The field is initially identified using keyword
    matching against tsow_knowledge.py.

    Its definition, guidance, alternate names and
    examples are then used for semantic retrieval.
    """

    parts = [
        "TSOW FIELD",
        field.get("question", ""),

        "STANDARD FIELD",
        field.get("standard_field", ""),

        "FIELD DEFINITION",
        field.get("description", ""),

        "MATCHING GUIDANCE",
        field.get("guidance", ""),

        "ALTERNATE FIELD NAMES",
        ", ".join(field.get("alternate_names", [])),

        "EXAMPLES",
        ", ".join(field.get("examples", [])),

        "FIELD RULES",
        field.get("rules", ""),

        "SUBFIELDS",
        ", ".join(field.get("subfields", []))
    ]

    return normalize_text(
        ". ".join(
            str(part)
            for part in parts
            if part
        )
    )


# ==================================================
# OLLAMA EMBEDDING CLIENT
# ==================================================

class OllamaEmbeddingClient:

    def __init__(
        self,
        base_url=OLLAMA_URL,
        model=EMBEDDING_MODEL
    ):

        self.base_url = base_url
        self.model = model

        self.session = requests.Session()

        self.lock = threading.RLock()

        print(
            f"[SEMANTIC] Ollama URL: {self.base_url}"
        )

        print(
            f"[SEMANTIC] Embedding model: {self.model}"
        )

    # --------------------------------------------------
    # SINGLE TEXT EMBEDDING
    # --------------------------------------------------

    def embed(self, text):
        """
        Generate one embedding using Ollama.
        """

        text = normalize_text(text)

        if not text:
            raise ValueError(
                "Cannot generate embedding for empty text."
            )

        payload = {
            "model": self.model,
            "input": text
        }

        try:

            response = self.session.post(
                f"{self.base_url}/api/embed",
                json=payload,
                timeout=EMBEDDING_TIMEOUT
            )

            response.raise_for_status()

            result = response.json()

            embeddings = result.get("embeddings", [])

            if not embeddings or not embeddings[0]:
                raise ValueError(
                    "Ollama returned an empty embedding."
                )

            vector = np.asarray(
                embeddings[0],
                dtype=np.float32
            )

            if not np.all(np.isfinite(vector)):
                raise ValueError(
                    "Embedding contains invalid numeric values."
                )

            return vector

        except requests.RequestException as e:

            raise RuntimeError(
                f"Ollama embedding request failed: {e}"
            ) from e

    # --------------------------------------------------
    # BATCH EMBEDDING
    # --------------------------------------------------

    def embed_batch(self, texts):
        """
        Generate embeddings for multiple texts
        in a single Ollama request.
        """

        texts = [
            normalize_text(text)
            for text in texts
            if normalize_text(text)
        ]

        if not texts:
            return []

        payload = {
            "model": self.model,
            "input": texts
        }

        try:

            response = self.session.post(
                f"{self.base_url}/api/embed",
                json=payload,
                timeout=EMBEDDING_TIMEOUT
            )

            response.raise_for_status()

            result = response.json()

            embeddings = result.get("embeddings", [])

            if len(embeddings) != len(texts):
                raise ValueError(
                    "Ollama embedding count mismatch."
                )

            vectors = []

            for embedding in embeddings:

                vector = np.asarray(
                    embedding,
                    dtype=np.float32
                )

                if not np.all(np.isfinite(vector)):
                    raise ValueError(
                        "Invalid embedding values."
                    )

                norm = np.linalg.norm(vector)

                if norm == 0:
                    raise ValueError(
                        "Zero-length embedding."
                    )

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

    # --------------------------------------------------
    # VERIFY OLLAMA MODEL
    # --------------------------------------------------

    def _check_model(self):
        """
        Verify Ollama is running and the configured
        embedding model is available.
        """

        try:

            response = self.client.session.get(
                f"{OLLAMA_URL}/api/tags",
                timeout=10
            )

            response.raise_for_status()

            models = response.json().get(
                "models", []
            )

            available = [
                model.get("name", "")
                for model in models
            ]

            model_available = any(
                name == self.client.model
                or name.split(":")[0]
                == self.client.model.split(":")[0]
                for name in available
            )

            if not model_available:

                raise RuntimeError(
                    f"Embedding model '{self.client.model}' "
                    f"not found in Ollama. "
                    f"Available models: {available}"
                )

            print(
                f"[SEMANTIC] Model verified: "
                f"{self.client.model}"
            )

        except requests.RequestException as e:

            raise RuntimeError(
                f"Cannot connect to Ollama at "
                f"{OLLAMA_URL}. Make sure Ollama is running. "
                f"Error: {e}"
            ) from e

    # --------------------------------------------------
    # EMBEDDING CACHE
    # --------------------------------------------------

    def get_embedding(self, text):
        """
        Retrieve a cached embedding or generate
        a new embedding.
        """

        text = normalize_text(text)

        if not text:
            return None

        cache_key = hashlib.sha256(
            (
                self.client.model + "|" + text
            ).encode("utf-8")
        ).hexdigest()

        with self.cache_lock:

            cached = self.embedding_cache.get(
                cache_key
            )

            if cached is not None:
                return cached

        embedding = self.client.embed(text)

        norm = np.linalg.norm(embedding)

        if norm == 0:
            raise ValueError(
                "Ollama returned a zero-length embedding."
            )

        embedding = embedding / norm

        with self.cache_lock:

            if len(self.embedding_cache) >= CACHE_SIZE:

                oldest_key = next(
                    iter(self.embedding_cache)
                )

                self.embedding_cache.pop(oldest_key)

            self.embedding_cache[cache_key] = embedding

        return embedding

    # --------------------------------------------------
    # CACHE-AWARE BATCH EMBEDDING
    # --------------------------------------------------

    def get_embeddings_batch(self, texts):
        """
        Generate embeddings in batches while reusing
        previously cached embeddings.

        The returned vectors preserve input order.
        """

        normalized = [
            normalize_text(text)
            for text in texts
        ]

        keys = [
            hashlib.sha256(
                (
                    self.client.model + "|" + text
                ).encode("utf-8")
            ).hexdigest()
            if text else None
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

            new_vectors = self.client.embed_batch(
                missing_texts
            )

            with self.cache_lock:

                for index, vector in zip(
                    missing_indices,
                    new_vectors
                ):

                    key = keys[index]

                    if len(self.embedding_cache) >= CACHE_SIZE:

                        oldest_key = next(
                            iter(self.embedding_cache)
                        )

                        self.embedding_cache.pop(oldest_key)

                    self.embedding_cache[key] = vector

                    results[index] = vector

        return results

    # --------------------------------------------------
    # COSINE SIMILARITY
    # --------------------------------------------------

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

        score = float(np.dot(emb1, emb2))

        return max(
            -1.0,
            min(1.0, score)
        )

    # --------------------------------------------------
    # FIELD EMBEDDING
    # --------------------------------------------------

    def field_embedding(self, field):
        """
        Generate the embedding for the complete
        identified TSOW field.
        """

        field_text = prepare_field_text(field)

        return self.get_embedding(
            "search_query: " + field_text
        )

    # --------------------------------------------------
    # QUESTIONNAIRE ROW RETRIEVAL
    # --------------------------------------------------

    def retrieve_rows(
        self,
        field,
        rows,
        top_k=TOP_K_ROWS
    ):
        """
        Compare a TSOW field definition against
        questionnaire question-answer rows.

        Returns:
            [(similarity_score, original_row), ...]
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

        row_vectors = self.get_embeddings_batch(
            valid_rows
        )

        scored_rows = []

        for row, vector in zip(
            valid_rows,
            row_vectors
        ):

            score = float(
                np.dot(field_vector, vector)
            )

            scored_rows.append(
                (score, row)
            )

        scored_rows.sort(
            key=lambda x: x[0],
            reverse=True
        )

        matched_rows = [
            (score, row)
            for score, row in scored_rows
            if score >= ROW_MATCH_THRESHOLD
        ]

        return matched_rows[:top_k]

    # --------------------------------------------------
    # SEMANTIC VALUE COMPARISON
    # --------------------------------------------------

    def values_agree(
        self,
        value1,
        value2,
        threshold=0.88
    ):
        """
        Compare two extracted values semantically.

        IMPORTANT:
        Semantic similarity does not guarantee
        technical equivalence.

        Do not use this method alone to resolve
        differences in model versions, processor
        generations, numerical values or specifications.
        """

        if not value1 or not value2:
            return False

        a = normalize_text(value1).casefold()
        b = normalize_text(value2).casefold()

        if a == b:
            return True

        score = self.similarity(
            value1,
            value2
        )

        return score >= threshold

    # --------------------------------------------------
    # WARM-UP
    # --------------------------------------------------

    def warm_up(self, texts):
        """
        Pre-generate embeddings for frequently
        used texts.
        """

        self.get_embeddings_batch(texts)

        print(
            f"[SEMANTIC] Warm-up completed. "
            f"Cached embeddings: "
            f"{len(self.embedding_cache)}"
        )

    # --------------------------------------------------
    # CLOSE
    # --------------------------------------------------

    def close(self):

        self.client.close()


# ==================================================
# SINGLETON INSTANCE
# ==================================================

_matcher = None

_matcher_lock = threading.Lock()


def get_matcher():
    """
    Return one shared semantic matcher.

    The Ollama client and embedding cache are reused
    across API requests.
    """

    global _matcher

    if _matcher is None:

        with _matcher_lock:

            if _matcher is None:

                _matcher = SemanticMatcher()

    return _matcher


# ==================================================
# PUBLIC HELPER FUNCTION
# ==================================================

def semantic_retrieve_rows(
    field,
    rows,
    top_k=TOP_K_ROWS
):
    """
    Convenience function for llm.py.

    Returns:
        [(similarity_score, row), ...]
    """

    matcher = get_matcher()

    return matcher.retrieve_rows(
        field,
        rows,
        top_k
    )


# ==================================================
# PUBLIC FUNCTION FOR LLM.PY
# ==================================================

def semantic_match_rows(field, rows, top_k=TOP_K_ROWS):
    """
    Public interface used by llm.py.

    Args:
        field: Identified TSOW field with its definition.
        rows: Questionnaire question-answer rows.
        top_k: Number of relevant rows to retrieve.

    Returns:
        List of relevant questionnaire rows.
    """

    matcher = get_matcher()

    matched_rows = matcher.retrieve_rows(
        field=field,
        rows=rows,
        top_k=top_k
    )

    print(
        f"[SEMANTIC] Field: {field.get('question')}"
    )

    for score, row in matched_rows:
        print(
            f"    Similarity: {score:.4f} | {row[:150]}"
        )

    return [
        row
        for score, row in matched_rows
    ]