import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

df = pd.read_csv('DATA/reviews_with_sentiment.csv')

N_TOPICS = 5
N_TOP_WORDS = 10

def remove_title_words(row):
    text = str(row['review/text']).lower()
    title_words=str(row['Title']).lower().split()
    for word in title_words:
        if len(word)>3:
            text = re.sub(r'\b'+re.escape(word)+r'\b','',text)
    return text

df['cleaned_text'] = df.apply(remove_title_words,axis=1)

def run_topic_modeling(text_series, label):
    print(f"\n{'='*60}")
    print(f"Topic Modeling for: {label} (n={len(text_series)})")
    print(f"{'='*60}")

    tfidf = TfidfVectorizer(
        max_df=0.9,        
        min_df=5,          
        stop_words='english',
        max_features=1000
    )
    tfidf_matrix = tfidf.fit_transform(text_series.astype(str))
    feature_names = tfidf.get_feature_names_out()

    nmf = NMF(n_components=N_TOPICS, random_state=42, init='nndsvda', max_iter=500)
    nmf.fit(tfidf_matrix)

    for topic_idx, topic in enumerate(nmf.components_):
        top_words = [feature_names[i] for i in topic.argsort()[-N_TOP_WORDS:][::-1]]
        print(f"\nTopic {topic_idx + 1}: {', '.join(top_words)}")

low_rating_reviews = df[df['review/score'].isin([1.0, 2.0])]['cleaned_text']
run_topic_modeling(low_rating_reviews, "Low Rating Reviews (1-2 stars)")

mid_rating_review = df[df['review/score'] == 3.0]['cleaned_text']
run_topic_modeling(mid_rating_review, "Mid Rating Reviews (3 stars)")



high_rating_reviews = df[df['review/score'].isin([4.0, 5.0])]['cleaned_text']
run_topic_modeling(high_rating_reviews, "High Rating Reviews (4-5 stars)")