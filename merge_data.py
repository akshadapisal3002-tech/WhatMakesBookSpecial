import pandas as pd
import ast

reviews = pd.read_csv('DATA/balance_sample.csv')

books_meta = pd.read_csv('DATA/books_data.csv' ,usecols=['Title','categories'])

merged_df = reviews.merge(books_meta,on='Title',how='left')



print("Merged shape:",merged_df.shape)
print("Missing genre count:",merged_df['categories'].isna().sum())
print(merged_df[['Title','review/score','categories']].head(10))

merged_df.to_csv('DATA/merge_reviews.csv',index=False)
print("Merged File Saved!")

merged_df = merged_df.dropna(subset=['categories'])
def clean_genre(val):
    try:
        genre_list = ast.literal_eval(val)  
        return genre_list[0] if len(genre_list) > 0 else None
    except:
        return None
merged_df['genre'] = merged_df['categories'].apply(clean_genre)
merged_df = merged_df.drop(columns=['categories'])

print("After cleaning shape:", merged_df.shape)
print(merged_df['genre'].value_counts().head(10))

merged_df.to_csv('DATA/merged_reviews_clean.csv', index=False)
print("Clean file saved!")

genre_counts = merged_df['genre'].value_counts()
THRESHOLD = 1000

valid_genres = genre_counts[genre_counts >= THRESHOLD].index.tolist()
print(f"\nGenres passing threshold (n >= {THRESHOLD}):")
print(valid_genres)
print(f"\nTotal valid genres: {len(valid_genres)}")

filtered_df = merged_df[merged_df['genre'].isin(valid_genres)]
print(f"\nShape before filtering: {merged_df.shape}")
print(f"Shape after filtering: {filtered_df.shape}")
print(filtered_df['genre'].value_counts())

filtered_df.to_csv('DATA/merged_reviews_final.csv', index=False)
print("Final filtered file saved!")