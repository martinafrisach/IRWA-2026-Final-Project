# Part 4 — RAG, User Interface and Web Analytics

## Overview

This folder contains the code and report for Part 4 of the IRWA Final Project, using the fashion products dataset.

It includes a Flask web application with a search page, a results page and a product details page, an LLM summary of the results (RAG), and a web analytics module with a dashboard.

- **GitHub URL:** https://github.com/martinafrisach/IRWA-2026-Final-Project.git
- **Tag:** `IRWA-2026-part-4`

## Files

- `web_app.py`: Flask application.
- `myapp/search/load_corpus.py`: Loads the dataset and builds the dictionary.
- `myapp/search/algorithms.py`: Text processing, index building and ranking (`build_terms`, `build_search_index`, `search_in_corpus`).
- `myapp/search/search_engine.py`: Class `SearchEngine` with the general `search` function called by the web app.
- `myapp/generation/rag.py`: Class `RAGGenerator`, which builds the prompt and calls the LLM.
- `myapp/analytics/analytics_data.py`: Class `AnalyticsData`, with the analytics tables, KPIs, charts, sessions and missions.
- `myapp/templates/`: HTML pages (home, results, document details, stats and dashboard).
- `IRWA-2026-u251664-u231662-part-4`: Detailed explanation of the decisions, results and conclusions.

## Dataset

The dataset is the same as in the previous parts (`fashion_products_dataset.json`, 28,080 fashion products). Its path is set with the variable `DATA_FILE_PATH` of the `.env` file, relative to `web_app.py`.

The corpus is loaded and the index is built once when the application starts, so every search is fast.

## Implemented Tasks

### Search engine
- One inverted index per field (title, description, category, brand, seller and product details), built with `build_search_index`.
- Conjunctive (AND) search: every returned document contains all the query terms in at least one field.
- BM25 ranking with a weight for each field (`WEIGHTS`), combined with a small boost for rating and discount and a penalty for out-of-stock products.
- General function `SearchEngine.search(search_query, search_id, corpus, top_k)`, called when the search button is clicked.

### User interface
- Search page with a central search box and a button.
- Results page with title, description, price, discount, rating, brand, stock status and a link to the original website, in ranking order.
- Document details page with the whole information of the product.

### RAG
- The top 5 results are sent to an LLM (Groq API) to generate a recommendation.
- Three improvements over the baseline: richer product information in the prompt (price, discount, rating, brand, stock and description), handling of the "no good products" case, and a stricter prompt with a lower temperature (0.2).

### Web analytics
- In-memory star schema with the tables `sessions`, `requests`, `queries` and `clicks`.
- Data collected: HTTP requests, sessions (browser, operating system, device, IP), queries (number of terms, number of results, order), clicks (query, rank) and dwell time.
- Dashboard (`/dashboard`) with KPIs, charts with explanations and the queries with zero results.
- Bonus: physical sessions (30 minutes of inactivity) and missions (consecutive queries sharing terms).

## How to Change the Options

- **Field weights, BM25 parameters and boosts:** edit `WEIGHTS`, `K1`, `B`, `ALPHA`, `BETA` and `GAMMA` at the top of `algorithms.py`. A weight of 0 keeps the field in the AND match but removes it from the score.
- **Number of results:** `top_k` in `SearchEngine.search`.
- **Number of products sent to the LLM:** `top_N` in `RAGGenerator.generate_response`.
- **LLM model:** variable `GROQ_MODEL` of the `.env` file.
- **Session timeout:** `SESSION_TIMEOUT` in `analytics_data.py`.

## How to Reproduce the Results

1. Clone the project repository.
2. Install the dependencies from the root `requirements.txt` (flask, pandas, numpy, nltk, pydantic, httpagentparser, altair, groq, python-dotenv).
3. Make sure `fashion_products_dataset.json` is located in the `data/` folder.
4. Create a `.env` file in the folder of `web_app.py`:

```
SECRET_KEY=any-random-string
SESSION_COOKIE_NAME=irwa_session
DATA_FILE_PATH=data/fashion_products_dataset.json
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b
DEBUG=True
```

5. Run `python web_app.py` and open http://localhost:8088 in the browser.
6. Search for some queries (for example `women full sleeve sweatshirt cotton` or `men slim jeans blue`), open some products.

Without a valid `GROQ_API_KEY` the search still works, but the summary box shows a message saying that RAG is not available. The analytics data is stored in memory, so it is lost when the server stops.

## Report: IRWA-2026-u251664-u231662-part-4

The PDF report provides a more detailed discussion of the search optimizations, the RAG improvements, the analytics data model and the dashboard.

## AI Usage
We used AI to review the report draft.
