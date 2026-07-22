import pandas as pd

df = pd.read_csv('DATA/reviews_with_sentiment.csv')

logistics_keywords = ['late', 'delivery', 'shipping', 'damaged', 'torn',
                      'packaging', 'arrived', 'shipped', 'package', 'box']

def has_logistics_complaint(text):
    text = str(text).lower()
    return any(keyword in text for keyword in logistics_keywords)

df['logistics_complaint'] = df['review/text'].apply(has_logistics_complaint)

low_rating_content = df[(df['review/score'] == 1.0) & (df['logistics_complaint'] == False)]

sample = low_rating_content.sample(10, random_state=42)

for i, row in sample.iterrows():
    print(f"\n{'='*60}")
    print(f"Rating: {row['review/score']} | Sentiment: {row['sentiment']:.3f}")
    print(f"Review: {row['review/text'][:400]}")  # पहिले 400 characters