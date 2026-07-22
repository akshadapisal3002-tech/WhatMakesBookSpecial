import pandas as pd

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

df = pd.read_csv('DATA/merged_reviews_final.csv')

genres_to_exclude =['Book burning','Dragons']
df = df[~df['genre'].isin(genres_to_exclude)]

analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    score = analyzer.polarity_scores(str(text))
    return score['compound']

df['sentiment']= df['review/text'].apply(get_sentiment)

print("Sentiment stats:")
print(df['sentiment'].describe())

correlation =df['sentiment'].corr(df['review/score'])
print(f"\nCorrelation between sentiment and rating:{correlation:.4f}")

print("\nAverage sentiment by rating:")
print(df.groupby('review/score')['sentiment'].mean())

df.to_csv('DATA/reviews_with_sentiment.csv', index=False)
print("\nSaved with sentinent sores!")