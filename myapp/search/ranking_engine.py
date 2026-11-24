"""
Ranking engine used by the web application (Part 4).

It packages the work of Parts 1-3 (pre-processing, per-field inverted indexes, TF-IDF,
BM25 and our own score) in a single class that is built ONCE when the web app starts,
instead of being re-run from the notebooks.
"""
import json
import math
import os
import pickle
import re
from collections import Counter

import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords", quiet=True)

# ----------------------------------------------------------------------------------
# Part 1: text pre-processing (same steps as build_terms of part1.ipynb)
# ----------------------------------------------------------------------------------
_stemmer = PorterStemmer()
_stop_words = set(stopwords.words("english"))
_stem_cache = {}   # the vocabulary is small compared to the number of tokens: stem each word once


def _stem(word):
    stem = _stem_cache.get(word)
    if stem is None:
        stem = _stem_cache[word] = _stemmer.stem(word)
    return stem


def build_terms(line):
    """
    Pre-process a text: lowercase, remove punctuation, tokenize, remove stop words, stem.

    Argument:
    line -- string (text) to be pre-processed

    Returns:
    list of tokens (empty list for missing values)
    """
    if not isinstance(line, str):
        return []
    line = re.sub(r"[^a-z0-9\s]", " ", line.lower()).split()
    return [_stem(w) for w in line if w not in _stop_words]


def details_to_text(details):
    """Join the values of product_details (list of one-key dicts) into one string."""
    text = []
    if isinstance(details, list):
        for d in details:
            if isinstance(d, dict):
                text.extend(str(v) for v in d.values())
    return " ".join(text)


# ----------------------------------------------------------------------------------
# Part 2/3 settings (same values as in the notebooks)
# ----------------------------------------------------------------------------------
FIELDS = ["title", "description", "category", "brand", "seller", "details"]
WEIGHTS = {"title": 3.0, "description": 1.5, "category": 0.5, "brand": 1.0, "seller": 0.0, "details": 2.0}
K1, B = 1.2, 0.75                 # BM25
ALPHA, BETA, GAMMA = 0.2, 0.1, 0.6  # own score: rating boost, discount boost, out-of-stock penalty
ALGORITHMS = ("custom", "bm25", "tfidf")


class SearchOutput:
    """Result of RankingEngine.search()."""

    def __init__(self, query_terms, pids, scores, matched, total, relaxed):
        self.query_terms = query_terms   # pre-processed query terms
        self.pids = pids                 # ranked list of pids (top_k)
        self.scores = scores             # score of each returned pid
        self.matched = matched           # how many query terms each returned pid contains
        self.total = total               # number of candidate documents (before cutting to top_k)
        self.relaxed = relaxed           # True if the strict AND gave too few results and it was relaxed


class RankingEngine:
    """Per-field inverted indexes + ranking functions, built once and kept in memory."""

    def __init__(self, data_path, cache_path=None):
        with open(data_path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        self.pids = [rec["pid"] for rec in data]
        self.N = len(self.pids)
        self._load_numeric_features(data)

        # Optimisation: the index is saved to disk and reused while the dataset does not change.
        stamp = (os.path.getmtime(data_path), self.N)
        cached = None
        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as fp:
                    cached = pickle.load(fp)
            except Exception:
                cached = None
        if cached and cached.get("stamp") == stamp:
            self.index, self.field_len = cached["index"], cached["field_len"]
        else:
            self.index, self.field_len = self._build_index(data)
            if cache_path:
                with open(cache_path, "wb") as fp:
                    pickle.dump({"stamp": stamp, "index": self.index, "field_len": self.field_len}, fp)

        self.avg_len = {f: max(self.field_len[f].mean(), 1e-9) for f in FIELDS}
        self._tfidf = None        # computed the first time TF-IDF is used
        self._cache = {}          # query cache

    # ------------------------------------------------------------------ building
    @staticmethod
    def _number(value):
        match = re.search(r"\d[\d,]*\.?\d*", str(value))
        return float(match.group().replace(",", "")) if match else np.nan

    def _load_numeric_features(self, data):
        """Numeric fields used by the own score and the filters (missing -> median / NaN)."""
        rating = pd.Series([self._number(r.get("average_rating")) for r in data])
        discount = pd.Series([self._number(r.get("discount")) for r in data])
        self.rating = (rating.fillna(rating.median()) / 5).clip(0, 1).to_numpy()
        self.discount = (discount.fillna(discount.median()) / 100).clip(0, 1).to_numpy()
        self.in_stock = np.array([not r.get("out_of_stock") for r in data])
        self.price = np.array([self._number(r.get("selling_price")) for r in data])        # NaN = unknown
        self.raw_rating = rating.to_numpy()                                                   # NaN = unknown

    def _build_index(self, data):
        """index[field][term] = {doc_id: term frequency}; field_len[field][doc_id] = number of terms."""
        index = {f: {} for f in FIELDS}
        field_len = {f: np.zeros(self.N) for f in FIELDS}
        for doc_id, rec in enumerate(data):
            texts = {
                "title": rec.get("title"),
                "description": rec.get("description"),
                "category": "{} {}".format(rec.get("category") or "", rec.get("sub_category") or ""),
                "brand": rec.get("brand"),
                "seller": rec.get("seller"),
                "details": details_to_text(rec.get("product_details")),
            }
            for f, text in texts.items():
                terms = build_terms(text)
                field_len[f][doc_id] = len(terms)
                for t, tf in Counter(terms).items():
                    index[f].setdefault(t, {})[doc_id] = tf
        return index, field_len

    # ------------------------------------------------------------------ retrieval
    def _candidates(self, q_terms, min_results):
        """
        Conjunctive (AND) retrieval. If it gives fewer than min_results documents, it is relaxed:
        any document with at least half of the query terms is accepted.

        Returns (docs, term_docs, matched, relaxed)
        """
        term_docs = {t: set() for t in q_terms}
        for t in q_terms:
            for f in FIELDS:
                term_docs[t].update(self.index[f].get(t, ()))
        docs = set.intersection(*term_docs.values())
        relaxed = False
        if len(docs) < min_results and len(q_terms) > 1:
            counts = Counter(d for s in term_docs.values() for d in s)
            need = max(1, math.ceil(len(q_terms) / 2))
            docs = {d for d, c in counts.items() if c >= need}
            relaxed = True
        matched = {d: sum(d in s for s in term_docs.values()) for d in docs}
        return docs, term_docs, matched, relaxed

    # ------------------------------------------------------------------ ranking
    def _bm25_scores(self, q_terms, docs, term_docs, k1=K1, b=B):
        idf = {t: math.log((self.N - len(term_docs[t]) + 0.5) / (len(term_docs[t]) + 0.5) + 1) for t in q_terms}
        scores = {}
        for d in docs:
            s = 0.0
            for t in q_terms:
                tf_tilde = 0.0
                for f in FIELDS:
                    if WEIGHTS[f] <= 0:
                        continue
                    tf = self.index[f].get(t, {}).get(d, 0)
                    if tf:
                        tf_tilde += WEIGHTS[f] * tf / (1 - b + b * self.field_len[f][d] / self.avg_len[f])
                if tf_tilde:
                    s += idf[t] * tf_tilde * (k1 + 1) / (tf_tilde + k1)
            scores[d] = s
        return scores

    def _ensure_tfidf(self):
        """idf and document-vector norms for TF-IDF + cosine (computed once, on first use)."""
        if self._tfidf is not None:
            return
        df_sets = {}
        for f in FIELDS:
            for t, postings in self.index[f].items():
                df_sets.setdefault(t, set()).update(postings)
        idf = {t: math.log(self.N / len(s)) for t, s in df_sets.items()}
        vecs = {}
        for f in FIELDS:
            for t, postings in self.index[f].items():
                for d, tf in postings.items():
                    vec = vecs.setdefault(d, {})
                    vec[t] = vec.get(t, 0.0) + WEIGHTS[f] * (1 + math.log(tf)) * idf[t]
        norms = np.zeros(self.N)
        for d, vec in vecs.items():
            norms[d] = math.sqrt(sum(v * v for v in vec.values()))
        self._tfidf = (idf, norms)

    def _tfidf_scores(self, q_terms, docs):
        self._ensure_tfidf()
        idf, norms = self._tfidf
        q = {t: idf.get(t, 0.0) for t in q_terms}          # tf = 1 for every unique query term
        q_norm = math.sqrt(sum(v * v for v in q.values()))
        scores = {}
        for d in docs:
            dot = 0.0
            for t in q_terms:
                w = sum(WEIGHTS[f] * (1 + math.log(self.index[f][t][d]))
                        for f in FIELDS if d in self.index[f].get(t, ()))
                dot += q[t] * w * idf.get(t, 0.0)
            scores[d] = dot / (q_norm * norms[d]) if q_norm and norms[d] else 0.0
        return scores

    # ------------------------------------------------------------------ public search
    def search(self, query, algorithm="custom", top_k=20, min_results=5,
               in_stock_only=False, max_price=None, min_rating=None):
        """
        Search the corpus.

        Arguments:
        query         -- text typed by the user
        algorithm     -- "custom" (BM25 + rating/discount/stock), "bm25" or "tfidf"
        top_k         -- number of documents returned
        min_results   -- if AND finds fewer documents than this, the search is relaxed (see _candidates)
        in_stock_only -- keep only products that are in stock
        max_price     -- keep only products with selling_price <= max_price
        min_rating    -- keep only products with average_rating >= min_rating

        Returns:
        SearchOutput
        """
        if algorithm not in ALGORITHMS:
            raise ValueError("algorithm must be one of {}".format(ALGORITHMS))
        key = (query, algorithm, top_k, min_results, in_stock_only, max_price, min_rating)
        if key in self._cache:
            return self._cache[key]

        q_terms = list(dict.fromkeys(build_terms(query)))          # unique terms, keeps order
        if not q_terms:
            return SearchOutput([], [], [], [], 0, False)

        docs, term_docs, matched, relaxed = self._candidates(q_terms, min_results)

        # filters
        if in_stock_only:
            docs = {d for d in docs if self.in_stock[d]}
        if max_price is not None:
            docs = {d for d in docs if self.price[d] <= max_price}      # unknown price (NaN) is dropped
        if min_rating is not None:
            docs = {d for d in docs if self.raw_rating[d] >= min_rating}

        if algorithm == "tfidf":
            scores = self._tfidf_scores(q_terms, docs)
        else:
            scores = self._bm25_scores(q_terms, docs, term_docs)
            if algorithm == "custom" and scores:
                top = max(scores.values()) or 1.0
                for d in scores:
                    boost = 1 + ALPHA * self.rating[d] + BETA * self.discount[d]
                    penalty = 1.0 if self.in_stock[d] else GAMMA
                    scores[d] = scores[d] / top * boost * penalty

        # documents with more query terms first (only matters when the search was relaxed),
        # then by score; ties are broken by doc id so the order is deterministic
        ranked = sorted(scores, key=lambda d: (-matched[d], -scores[d], d))[:top_k]
        out = SearchOutput(q_terms, [self.pids[d] for d in ranked], [float(scores[d]) for d in ranked],
                           [matched[d] for d in ranked], len(docs), relaxed)

        if len(self._cache) > 256:
            self._cache.clear()
        self._cache[key] = out
        return out