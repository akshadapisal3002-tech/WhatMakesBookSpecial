# 📚 What Makes a Good Book?

A data science project analyzing 40,000+ Amazon book reviews to understand what drives book ratings — using sentiment analysis, topic modeling, and genre-wise statistical comparison.

🔗 **GitHub Repo:** https://github.com/akshadapisal3002-tech/WhatMakesBookSpecial

---

## Project Overview

Star ratings alone don't tell the full story of why readers love or dislike a book. This project digs into the actual text of Amazon book reviews to answer:

- Does the emotional tone (sentiment) of a review actually match its star rating?
- Do different genres have different average ratings — and by how much?
- What specific topics/themes come up in low-rated vs. high-rated reviews?

## Dataset

- **Source:** [Amazon Books Reviews](https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews) (Kaggle) — 3M reviews across 212,404 books, originally compiled from Amazon review data (1996–2014) and Google Books API metadata.
- **Sampling:** A balanced sample of 15,000 reviews per star rating (1–5) was drawn using chunked reading, to avoid memory overload and rating-class imbalance.
- **Genre filtering:** Reviews were merged with book metadata to attach genre labels. Genres with fewer than 1,000 reviews were excluded to keep comparisons statistically meaningful. Two non-meaningful categories (mislabeled by the source API) were also removed.
- **Final dataset:** ~38,000 reviews across 8 genres (Fiction, History, Religion, Computers, Business & Economics, Biography & Autobiography, Education, Juvenile Fiction).

## Methodology

1. **Data Engineering:** Chunked CSV reading + balanced sampling to handle a 2.86GB raw file on limited RAM.
2. **Data Cleaning:** Genre extraction from nested string-lists, missing value handling, threshold-based genre filtering.
3. **Exploratory Data Analysis:** Rating distribution, review length distribution, genre-wise average ratings.
4. **Sentiment Analysis:** VADER (rule-based sentiment analyzer) used to score each review's text on a -1 (negative) to +1 (positive) scale.
5. **Hypothesis Testing:** Tested whether low ratings were driven by non-content complaints (e.g., shipping/packaging issues) rather than dissatisfaction with the book itself.
6. **Topic Modeling:** TF-IDF + NMF (Non-negative Matrix Factorization) applied separately to low, mid, and high-rated reviews to surface distinct themes.
7. **Dashboard:** Interactive Streamlit dashboard to explore all findings, filter by genre/rating, and read sample reviews.

## Key Findings

- **Genre matters a lot:** Average rating varies by nearly 1.7 points between the highest-rated genre (Education, 3.98) and the lowest (Computers, 2.29).
- **Sentiment and rating are only moderately correlated (r = 0.31):** Sentiment alone doesn't fully explain star ratings.
- **1-star reviews are surprisingly sentiment-neutral (avg. 0.04), not strongly negative** — suggesting low ratings are often driven by factual, non-emotional complaints rather than angry language.
- **Tested and largely ruled out shipping/logistics complaints as the cause** — only 13% of 1–2 star reviews mentioned delivery/packaging issues, and the sentiment difference was small. The gap between sentiment and rating is more likely due to VADER's difficulty detecting subtle disappointment and mixed sentiment.
- **Topic modeling** revealed recurring themes like pacing/boredom and value-for-money in low-rated reviews, versus emotional connection and strong recommendation language in high-rated reviews — with a known limitation that some topics reflect specific popular titles/authors rather than pure generic themes.

## Tech Stack

- **Data Processing:** Python, Pandas, NumPy
- **NLP:** VADER Sentiment, Scikit-learn (TF-IDF, NMF)
- **Visualization/Dashboard:** Streamlit, Plotly
- **Version Control:** Git, GitHub

## How to Run Locally

```bash
# Clone the repo
git clone https://github.com/akshadapisal3002-tech/WhatMakesBookSpecial.git
cd WhatMakesBookSpecial

# Install dependencies
pip install pandas numpy nltk vaderSentiment scikit-learn matplotlib seaborn streamlit plotly

# Run the dashboard
streamlit run dashboard.py
```

Note: The raw dataset files are not included in this repo due to size (2.86GB). Download `Books_rating.csv` and `books_data.csv` from the [Kaggle dataset page](https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews) and place them in a `DATA/` folder, then run the pipeline scripts in order:
1. `data_file.py` — creates the balanced sample
2. `merge_data.py` — merges genre metadata and applies filtering
3. `eda.py` — exploratory analysis
4. `sentiment.py` — sentiment scoring
5. `Topic_modeling.py` — topic modeling
6. `dashboard.py` — launches the interactive dashboard

## Limitations & Future Work

- Topic modeling occasionally surfaces specific book/author names instead of pure generic themes — could be improved with Named Entity Recognition to filter proper nouns more comprehensively.
- VADER is a rule-based sentiment tool and may miss sarcasm, subtle disappointment, or mixed sentiment — a transformer-based sentiment model (e.g., DistilBERT) could improve accuracy.
- Genre labels come from Google Books API categorization, which is not always a clean, reader-facing genre taxonomy.

## Author

Akshada Pisal — [LinkedIn](https://www.linkedin.com/in/akshadapisal/) | [GitHub](https://github.com/akshadapisal3002-tech)