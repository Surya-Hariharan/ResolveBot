"""Policy search: embed the chunks once, rank them by cosine similarity.

Vectors are normalized, so cosine similarity is just a dot product. If the
embedding model cannot be loaded (no network the first time, for example) the
search falls back to simple keyword matching so the service keeps working.
"""
import math
import re
import threading
import time

from app.retrieval import policy_loader

# Models that expect a prefix on the query and/or the passage (from their model cards)
PREFIXES = {
    "e5": {"query": "query: ", "passage": "passage: "},
    "bge": {"query": "Represent this sentence for searching relevant passages: ", "passage": ""},
}
STOP_WORDS = {
    "a", "an", "the", "is", "are", "do", "does", "i", "my", "me", "you", "your", "to", "of", "for",
    "in", "on", "it", "can", "how", "what", "when", "if", "and", "or", "we", "our", "be", "get",
}
WARMUP_WAIT_SECONDS = 10  # how long the first search waits for the model before using keywords


def prefixes_for(model_name):
    for key, value in PREFIXES.items():
        if key in model_name.lower():
            return value
    return {"query": "", "passage": ""}


def words_of(text):
    return [w for w in re.findall(r"[a-z0-9₹]+", text.lower()) if w not in STOP_WORDS]


def stem(word):
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


class PolicySearch:
    def __init__(self, settings=None, chunks=None, model_name=None, top_k=None, min_score=None, today=None):
        # Either pass settings (normal use) or explicit values (experiments and tests)
        self.model_name = model_name or settings.embedding_model
        self.top_k = top_k or settings.retrieval_top_k
        self.min_score = settings.retrieval_min_score if min_score is None else min_score
        if chunks is None:
            chunks = policy_loader.load_chunks(settings.policies_path, settings.manifest_path, settings.app_today)
        self.chunks = chunks
        self.support_email = policy_loader.find_support_email(chunks)
        self._model = None
        self._vectors = None
        self._ready = threading.Event()  # set when the embedding model is loaded (or failed)
        self._lock = threading.Lock()
        self.load_error = None
        self._keyword_words = [[stem(w) for w in words_of(c.embed_text())] for c in chunks]

    # ------------------------------------------------------------ setup
    def warm_up(self):
        """Load the model and embed every chunk. Safe to call from a background thread."""
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(self.model_name)
            prefix = prefixes_for(self.model_name)["passage"]
            vectors = model.encode(
                [prefix + c.embed_text() for c in self.chunks],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            self._model, self._vectors = model, vectors
        except Exception as problem:  # model download or install trouble: keep running on keywords
            self.load_error = f"{type(problem).__name__}: {problem}"
        finally:
            self._ready.set()

    def reindex(self, chunks):
        """Embed a different set of chunks with the model that is already loaded (experiments)."""
        self.chunks = chunks
        self._keyword_words = [[stem(w) for w in words_of(c.embed_text())] for c in chunks]
        prefix = prefixes_for(self.model_name)["passage"]
        self._vectors = self._model.encode(
            [prefix + c.embed_text() for c in chunks], normalize_embeddings=True, show_progress_bar=False
        )

    def start_warm_up_in_background(self):
        threading.Thread(target=self.warm_up, daemon=True).start()

    @property
    def method(self):
        return "dense" if self._vectors is not None else "keyword"

    # ---------------------------------------------------------- ranking
    def rank(self, query, extra_query=None):
        """All chunks with a score, best first. No filtering by score.

        extra_query (the customer's own message) is searched too, and each chunk keeps
        its better score, so a terse query written by the LLM cannot hide a good match.
        """
        self._ready.wait(timeout=WARMUP_WAIT_SECONDS)
        ranked = self._rank_one(query)
        if extra_query and extra_query.strip() and extra_query.strip() != query:
            best = {id(chunk): score for chunk, score in ranked}
            for chunk, score in self._rank_one(extra_query.strip()):
                best[id(chunk)] = max(best[id(chunk)], score)
            ranked = sorted(((chunk, best[id(chunk)]) for chunk, _ in ranked), key=lambda pair: -pair[1])
        return ranked

    def _rank_one(self, query):
        if self._vectors is not None:
            return self._rank_dense(query)
        return self._rank_keywords(query)

    def _rank_dense(self, query):
        prefix = prefixes_for(self.model_name)["query"]
        with self._lock:
            query_vector = self._model.encode([prefix + query], normalize_embeddings=True, show_progress_bar=False)[0]
        scores = self._vectors @ query_vector
        order = sorted(range(len(self.chunks)), key=lambda i: -float(scores[i]))
        return [(self.chunks[i], float(scores[i])) for i in order]

    def _rank_keywords(self, query):
        query_words = {stem(w) for w in words_of(query)}
        document_count = len(self.chunks)
        frequency = {}
        for words in self._keyword_words:
            for word in set(words):
                frequency[word] = frequency.get(word, 0) + 1
        scored = []
        for chunk, words in zip(self.chunks, self._keyword_words):
            matched = query_words & set(words)
            weight = sum(math.log(1 + document_count / frequency[w]) for w in matched)
            best_possible = sum(math.log(1 + document_count / frequency.get(w, 1)) for w in query_words) or 1
            scored.append((chunk, weight / best_possible))
        return sorted(scored, key=lambda pair: -pair[1])

    # ----------------------------------------------------------- search
    def search(self, query, extra_query=None):
        """The paragraphs to show the LLM: top-k in force, plus a future change if relevant."""
        ranked = self.rank(query, extra_query)
        floor = self.min_score if self.method == "dense" else 0.35
        in_force = [(c, s) for c, s in ranked if c.in_force and s >= floor][: self.top_k]
        future = [(c, s) for c, s in ranked if not c.in_force and s >= floor][:1]
        picked = in_force + future
        # Policy text goes before FAQ text, because the policy wins when they disagree
        picked.sort(key=lambda pair: pair[0].doc_type == "faq")
        return [self._as_paragraph(chunk, score) for chunk, score in picked]

    @staticmethod
    def _as_paragraph(chunk, score):
        paragraph = {
            "doc_id": chunk.doc_id,
            "section": chunk.section,
            "title": chunk.doc_title,
            "text": chunk.text,
            "score": round(score, 3),
        }
        if not chunk.in_force:
            from app.rules import display_date

            paragraph["note"] = (
                f"NOT IN FORCE YET. This starts on {display_date(chunk.effective_date)}. "
                "Mention it only as a future change with that date."
            )
        elif chunk.doc_type == "faq":
            paragraph["note"] = "FAQ summary. If it disagrees with a policy paragraph, the policy wins."
        return paragraph


def timed(function, *args):
    """Run a function and return (result, seconds). Used by the experiments."""
    start = time.perf_counter()
    result = function(*args)
    return result, time.perf_counter() - start
