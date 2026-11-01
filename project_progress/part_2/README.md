# Part 2 — Indexing and Evaluation

## Overview

This folder contains the code and report for Part 2 of the IRWA Final Project, using the fashion products dataset.

It includes the construction of the inverted index, conjunctive (AND) search with TF-IDF ranking, the definition of five test queries, and the evaluation of the search engine with ranking metrics.

- **GitHub URL:** https://github.com/martinafrisach/IRWA-2026-Final-Project.git
- **Tag:** `IRWA-2026-part-2`

## Files

- `part2.ipynb`: Notebook which has the indexing, search, ranking and evaluation code.
- `README.md`: Description of the work and instructions to reproduce the results.
- `IRWA-2026-u251664-u231662-part-2`: Detailed explanation of the decisions, results and conclusions.

## Dataset

The dataset is located at `../../data/fashion_products_dataset.json` and contains 28,080 fashion products.
The file `../../data/validation_labels.csv` contains the relevance labels of the two predefined queries and is used to evaluate the search engine.

The notebook starts by running `../part_1/part1.ipynb`, which loads the dataset and defines the pre-processing functions (`build_terms`, `details_to_text`) and the term columns of each field.

## Implemented Tasks

### Indexing
- One inverted index per field (title, description, category, brand, seller and product details) with the function `create_index`. Each term stores the list of documents where it appears and its positions.
- Hybrid field handling decided in Part 1: category and sub_category are merged, and the rest of the fields are kept separate.
- Conjunctive (AND) search with the function `search`: every returned document contains all the query terms, in at least one field.
- Five test queries chosen from the most frequent terms of the index.

### Ranking
- TF-IDF ranking with a weight for each field (`weights`).
- Ranked results for the test queries with the function `show`.

### Evaluation
- Metrics: Precision@K, Recall@K, F1@K, Average Precision@K, MAP, MRR and NDCG@K, with sanity checks.
- Evaluation on the two predefined queries of `validation_labels.csv` with the function `get_evaluation_data`.
- Relevance labels assigned by us for our five test queries, and the evaluation of their results.

## How to Change the Options

- **Field weights:** edit the `weights` dictionary. A weight of 0 keeps the field in the AND match but removes it from the score.
- **Queries:** edit the `test_queries` dictionary and run the cells below it again.
- **Number of results and cut-off:** `top_k` in `search` and `k` in the metric functions and in `get_evaluation_data`.

## How to Reproduce the Results

1. Clone the project repository.
2. Install the dependencies from the root `requirements.txt` (numpy, pandas, matplotlib, nltk, jupyter) and scikit-learn, which is used for the sanity checks of the metrics.
3. Make sure `fashion_products_dataset.json` and `validation_labels.csv` are located in the `data/` folder.
4. Open `part2.ipynb` using Jupyter Notebook or VS Code, from the `part_2` folder (Part 1 must be in `part_1`).
5. Run all cells in order (Restart & Run All).

## Report: IRWA-2026-u251664-u231662-part-2

The PDF report provides a more detailed discussion of the indexing decisions, the ranking, the choice of the test queries and the evaluation results.

## AI Usage
We used AI to review the report draft and debug some code.
