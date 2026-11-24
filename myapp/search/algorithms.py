# myapp/search/algorithms.py
import math
import re
from array import array
from collections import defaultdict
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("stopwords", quiet=True)
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

WEIGHTS = {"title": 3.0, "description": 1.5, "category": 0.5, "brand": 1.0, "seller": 0.0, "details": 2.0}
K1, B = 1.2, 0.75
ALPHA, BETA, GAMMA = 0.2, 0.1, 0.6


def build_terms(line):
    if not isinstance(line, str):
        return []
    line = line.lower()
    line = re.sub(r"[^a-z0-9\s]", " ", line)
    tokens = line.split()
    return [stemmer.stem(w) for w in tokens if w not in stop_words]


def details_to_text(details):
    if isinstance(details, dict):
        return " ".join(str(v) for v in details.values())
    return ""


def build_search_index(corpus: dict):
    """
    Call this ONCE when the app starts (e.g., inside load_corpus.py).
    Converts the Dict[pid, Document] into inverted indexes and precomputed stats.
    """
    pids = list(corpus.keys())
    N = len(pids)

    field_terms = {f: [] for f in WEIGHTS}
    for pid in pids:
        doc = corpus[pid]
        field_terms["title"].append(build_terms(doc.title))
        field_terms["description"].append(build_terms(doc.description))
        cat_str = f"{doc.category or ''} {doc.sub_category or ''}"
        field_terms["category"].append(build_terms(cat_str))
        field_terms["brand"].append(build_terms(doc.brand))
        field_terms["seller"].append(build_terms(doc.seller))
        field_terms["details"].append(build_terms(details_to_text(doc.product_details)))

    # Build inverted index per field
    index = {}
    for field, all_terms in field_terms.items():
        field_idx = defaultdict(list)
        for doc_id, terms in enumerate(all_terms):
            curr = {}
            for pos, term in enumerate(terms):
                if term in curr:
                    curr[term][1].append(pos)
                else:
                    curr[term] = [doc_id, array("I", [pos])]
            for term, posting in curr.items():
                field_idx[term].append(posting)
        index[field] = field_idx

    # Precompute BM25 field lengths
    field_len = {f: np.array([len(t) for t in field_terms[f]], dtype=float) for f in field_terms}
    avg_len = {f: max(field_len[f].mean(), 1e-9) for f in field_len}

    # Precompute normalized product features for custom ranking
    ratings = np.array([doc.average_rating if doc.average_rating is not None else 3.5 for doc in corpus.values()]) / 5.0
    discounts = np.array([doc.discount if doc.discount is not None else 0.0 for doc in corpus.values()]) / 100.0
    in_stock = np.array([not doc.out_of_stock for doc in corpus.values()])

    return {
        "pids": pids,
        "N": N,
        "index": index,
        "field_len": field_len,
        "avg_len": avg_len,
        "ratings": np.clip(ratings, 0, 1),
        "discounts": np.clip(discounts, 0, 1),
        "in_stock": in_stock,
    }


def search_in_corpus(query: str, index_data: dict, top_k: int = 20):
    """
    Runs conjunctive retrieval + Custom BM25 ranking.
    Returns a list of tuples: [(pid, score), ...]
    """
    q_terms = list(dict.fromkeys(build_terms(query)))
    if not q_terms:
        return []

    index = index_data["index"]
    pids = index_data["pids"]
    N = index_data["N"]
    field_len = index_data["field_len"]
    avg_len = index_data["avg_len"]

    postings, term_docs = {}, {t: set() for t in q_terms}
    for t in q_terms:
        for field in index:
            tf = {}
            for doc_id, positions in index[field].get(t, []):
                tf[doc_id] = len(positions)
                term_docs[t].add(doc_id)
            postings[(field, t)] = tf

    docs = set.intersection(*term_docs.values())
    if not docs:
        return []

    # Compute BM25 scores
    bm25 = {}
    for d in docs:
        s = 0.0
        for t in q_terms:
            tf_tilde = 0.0
            for f in index:
                tf = postings[(f, t)].get(d, 0)
                if tf and WEIGHTS[f] > 0:
                    tf_tilde += WEIGHTS[f] * tf / (1 - B + B * field_len[f][d] / avg_len[f])
            n = len(term_docs[t])
            idf = math.log((N - n + 0.5) / (n + 0.5) + 1)
            s += idf * tf_tilde * (K1 + 1) / (tf_tilde + K1)
        bm25[d] = s

    # Apply Custom Boosts (Rating, Discount, Stock)
    max_score = max(bm25.values()) or 1.0
    final_scores = {}
    for d, s in bm25.items():
        boost = 1 + ALPHA * index_data["ratings"][d] + BETA * index_data["discounts"][d]
        penalty = GAMMA if not index_data["in_stock"][d] else 1.0
        final_scores[d] = (s / max_score) * boost * penalty

    ranked = sorted(final_scores.items(), key=lambda x: (-x[1], x[0]))[:top_k]
    return [(pids[doc_id], score) for doc_id, score in ranked]