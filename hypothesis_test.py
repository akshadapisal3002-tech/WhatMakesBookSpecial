import pandas as pd

df = pd.read_csv('DATA/reviews_with_sentiment.csv')

logistics_keywords=['late','delivery','shipping','damaged','torn',
                    'packaging','arrived','shipped','package','box']

def has_logistics_complaint(text):
    text = str(text).lower()

    return any(keyword in text for keyword in logistics_keywords )

df['logistics_complaint']=df['review/text'].apply(has_logistics_complaint)

low_rating = df[df['review/score'].isin([1.0,2.0])]

print("Total 1-2 star reviews:",len(low_rating))
print("With logistics complaints:",low_rating['logistics_complaint'].sum())
print("Percentage:",(low_rating['logistics_complaint'].sum()/len(low_rating))*100,"%")

print("\nAverage sentiment compariosn(1-2 star reviews only):")
print(low_rating.groupby('logistics_complaint')['sentiment'].mean())
print("\nCount comparison:")
print(low_rating.groupby('logistics_complaint')['sentiment'].count())