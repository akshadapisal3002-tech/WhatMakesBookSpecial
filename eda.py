import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('DATA/merged_reviews_final.csv')


print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nMissing values:\n", df.isna().sum())

df['review_length'] = df['review/text'].astype(str).apply(lambda x: len(x.split()))


print("\nReview length stats:")
print(df['review_length'].describe())

genres_to_exclude = ['Book burning', 'Dragons']
df = df[~df['genre'].isin(genres_to_exclude)]

print(f"\nShape after excluding odd genres: {df.shape}")
print(df['genre'].value_counts())

genre_avg_rating = df.groupby('genre')['review/score'].mean().sort_values(ascending=False)
print("\nAverage rating by genre:")
print(genre_avg_rating)