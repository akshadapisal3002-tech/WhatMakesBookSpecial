import pandas as pd

CHUNK_SIZE = 10000
TARGET_PER_RATING = 15000
columns_needed = ['Title','review/score','review/summary','review/text','review/helpfulness']

samples ={1.0:[], 2.0:[],3.0:[],4.0:[],5.0:[]}
counts={1.0:0,2.0:0,3.0:0,4.0:0,5.0:0}

reader = pd.read_csv(
    'DATA/Books_rating.csv',
    usecols=columns_needed,
    chunksize=CHUNK_SIZE
)
for chunk in reader:
    chunk = chunk.dropna(subset=['review/score','review/text'])

    for rating in samples.keys():
        if counts[rating] >= TARGET_PER_RATING :
            continue
                  
        needed = TARGET_PER_RATING - counts[rating]

        matched = chunk[chunk['review/score']==rating].head(needed)
        samples[rating].append(matched)
        counts[rating]+=len(matched)
    
    if all(counts[r] >=TARGET_PER_RATING for r in counts):
        break
balanced_df = pd.concat([pd.concat(samples[r])for r in samples], ignore_index=True)
print(balanced_df['review/score'].value_counts())
print(balanced_df.shape)
balanced_df.to_csv('data/balance_sample.csv',index=False)
print("Saved Successfully!")

