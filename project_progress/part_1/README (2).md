# Part 1 — Data Preprocessing and Exploratory Data Analysis

## Overview

This folder contains the code and report for Part 1 of the IRWA Final Project, using the fashion products dataset.

It includes data loading, text preprocessing, and exploratory data analysis (EDA) to better understand the dataset before continung to do the search engine.

- **GitHub URL:** https://github.com/martinafrisach/IRWA-2026-Final-Project.git
- **Tag:** `IRWA-2026-part-1`

## Files

- `part1.ipynb`:  Notebook which has the preprocessing and exploratory analysis code.
- `README.md`: Description of the work and instructions to reproduce the results.
- `report` Detailed explanation of the results and conclusions.

## Dataset

The dataset is located at `../../data/fashion_products_dataset.json` and contains 28,080 fashion products.
The file `../../data/validation_labels.csv` is used to check which fields contain the terms of the two validation queries.

## Implemented Tasks

### Data Preprocessing
- Text normalization (lowercase) and tokenization.
- Punctuation removal, stopword removal and stemming  with the function `build_terms`.
- Conversion of numerical fields (`selling_price`, `actual_price`, `discount`, `average_rating`) with the function `parse_number`.
- Checking how the labels match each query

### Exploratory Data Analysis
- Dataset overview and data quality.
- Product categories, subcategories, brands, and sellers.
- Text length and vocabulary analysis.
- Most frequent terms.
- Price, discount, and rating distributions.
- Product rankings and stock availability.

## How to Reproduce the Results

1. Clone the project repository.
2. Install the dependencies from the root `requirements.txt` (numpy, pandas, matplotlib, nltk, jupyter).
3. Make sure `fashion_products_dataset.json` and `validation_labels.csv` are located in the `data/` folder.
4. Open `part1.ipynb` using  Notebook or VS Code.
5. Run all cells in order (Restart & Run All).


## Report

The PDF report provides a more detailed discussion of the preprocessing decisions and exploratory analysis.

## AI Usage

We used AI to review the report draft and debug some code.
