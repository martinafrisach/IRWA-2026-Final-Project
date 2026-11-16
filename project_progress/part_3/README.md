# Part 3 — Ranking and Filtering

## Overview

This folder contains the code and report for Part 3 of the IRWA Final Project, using the fashion products dataset.

It includes the conjunctive (AND) retrieval pipeline, three ranking functions (TF-IDF + cosine similarity, BM25 and our own score), a Word2Vec + cosine ranking, and the comparison of all of them with the ranking metrics of Part 2.

- **GitHub URL:** https://github.com/martinafrisach/IRWA-2026-Final-Project.git
- **Tag:** `IRWA-2026-part-3`

## Files

- `part3.ipynb`: Notebook which has the retrieval, the ranking functions, the Word2Vec model and the evaluation code.
- `README.md`: Description of the work and instructions to reproduce the results.
- `IRWA-2026-u251664-u231662-part-3`: Detailed explanation of the decisions, results and conclusions.

## Dataset

The dataset is located at `../../data/fashion_products_dataset.json` and contains 28,080 fashion products.
The file `../../data/validation_labels.csv` contains the relevance labels of the two predefined queries and is used to evaluate the rankings.

The notebook starts by running `../part_2/part2.ipynb` (which in turn runs `../part_1/part1.ipynb`). From them it reuses the inverted indexes (`index`), the term columns of each field (`field_terms`), the field `weights`, the function `build_terms`, the test and validation queries, and the evaluation metrics.

## Implemented Tasks

### Retrieval
- Conjunctive (AND) search with the function `get_candidates`: a document is a candidate only if it contains all the query terms, in at least one field. It also returns the term frequencies of every field, which the ranking functions need.
- All the ranking functions share this same filtering step, so they only differ in the order of the documents.
- `top_k_from_scores` sorts the scores and cuts the list to the top-k (ties are broken by document id, so the results are deterministic).

### Ranking
- **TF-IDF + cosine similarity** (`tf_idf` and `cosine_similarity`): TF-IDF vector of each document with the weight of each field, and cosine similarity with the query vector.
- **BM25** (`bm25_scores` and `rank_bm25`): BM25 with the term frequencies of the fields combined using the field weights and a length normalisation per field (`K1 = 1.2`, `B = 0.75`).
- **Our own score** (`rank_custom`): BM25 (normalised by the best score of the query) multiplied by a boost that uses the numerical fields: `(1 + ALPHA·rating + BETA·discount)`, and by a penalty `GAMMA` if the product is out of stock (`ALPHA = 0.2`, `BETA = 0.1`, `GAMMA = 0.6`). Missing ratings and discounts are filled with the median.
- **Word2Vec + cosine similarity** (`rank_word2vec`): Word2Vec trained on our pre-processed tokens (`gensim`, 100 dimensions). Each text (query or document) is the average of its word vectors, and the candidates of the AND search are ranked by cosine similarity. It returns the top-20 documents for each of the five test queries (`w2v_results`).

### Evaluation
- Function `evaluate_ranker`: it applies the metrics of Part 2 (Precision@10, Recall@10, F1@10, Average Precision@10, MAP, MRR and NDCG@10) to any ranking function, using the two predefined queries of `validation_labels.csv`.
- Function `show_ranking`: it prints the top results of each ranking function for a query, to compare how the order changes.

## How to Change the Options

- **Field weights:** edit the `weights` dictionary of `part2.ipynb`. It is used by TF-IDF and BM25 (and so by our own score). A weight of 0 keeps the field in the AND match but removes it from the score.
- **BM25 parameters:** edit `K1` and `B`, or pass `k1` and `b` to `rank_bm25`.
- **Own score:** edit `ALPHA` (rating), `BETA` (discount) and `GAMMA` (out-of-stock penalty).
- **Word2Vec:** edit the parameters of `Word2Vec(...)` (`vector_size`, `window`, `min_count`, `epochs`).
- **Queries:** edit the `test_queries` dictionary in `part2.ipynb`, or call any ranking function directly, e.g. `rank_bm25("men slim jeans blue", top_k=20)`.
- **Number of results and cut-off:** `top_k` in the ranking functions and `k` in `evaluate_ranker`.

## How to Reproduce the Results

1. Clone the project repository.
2. Install the dependencies from the root `requirements.txt` (numpy, pandas, matplotlib, nltk, jupyter, scikit-learn) and `gensim`, which is used for Word2Vec (`pip install gensim`, or run the cell `%pip install gensim` of the notebook).
3. Make sure `fashion_products_dataset.json` and `validation_labels.csv` are located in the `data/` folder.
4. Open `part3.ipynb` using Jupyter Notebook or VS Code, from the `part_3` folder (Part 1 must be in `part_1` and Part 2 in `part_2`).
5. Run all cells in order (Restart & Run All). The Word2Vec model uses a fixed seed and one worker, so the results are reproducible.

## Report: IRWA-2026-u251664-u231662-part-3

The PDF report provides a more detailed discussion of the differences between TF-IDF and BM25, the design of our own score (pros and cons), the Word2Vec results and the comparison with other representations (Doc2Vec and sentence embeddings).

## AI Usage
We used AI to review the report draft and to help with the code of the ranking functions.
