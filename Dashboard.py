import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="What Makes A Good Book?", layout="wide")
st.markdown("""
    <style>
    .stApp {
        background-color: #1d1716;
    }
    body, p, div, span, label {
        color: #FFFFFF;
    }
    [data-testid="stMetric"] {
        background-color: #402a23;
        border: 2px solid #a55233;
        border-radius: 10px;
        padding: 15px;
    }
    [data-testid="stMetricLabel"] {
        color: #FFFFFF;
    }
    [data-testid="stMetricValue"] {
        color: #FFFFFF;
    }
    h1 {
        color: #FFFFFF;
    }
    h2 {
        color: #FFFFFF;
        border-bottom: 2px solid #a55233;
        padding-bottom: 5px;
    }
    h3 {
        color: #FFFFFF;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #402a23;
        border-radius: 5px;
        padding: 8px 16px;
        color: #FFFFFF;
    }
    .stTabs [aria-selected="true"] {
        background-color: #a55233;
        color: #FFFFFF;
    }
    [data-testid="stExpander"] {
        background-color: #402a23;
        border: 1px solid #a55233;
        color: #FFFFFF;
    }
    </style>
""", unsafe_allow_html=True)
@st.cache_data
def load_data():
    df = pd.read_csv('DATA/reviews_with_sentiment.csv')
    genres_to_exclude = ['Book burning', 'Dragons']
    df = df[~df['genre'].isin(genres_to_exclude)]
    return df

df = load_data()

st.title("What Makes a Good Book?")
st.markdown("Sentiment, Topic, and Genre-based analysis of Amazon Book Reviews")
st.header("Overview")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Reviews", f"{len(df):,}")
col2.metric("Genres Analyzed", df['genre'].nunique())
col3.metric("Average Rating", f"{df['review/score'].mean():.2f}")
col4.metric("Average Sentiment", f"{df['sentiment'].mean():.2f}")

st.subheader("Rating Distribution")
rating_counts = df['review/score'].value_counts().sort_index()
fig_rating = px.bar(
    x=rating_counts.index,
    y=rating_counts.values,
    labels={'x': 'Rating', 'y': 'Number of Reviews'},
    title="How Many Reviews For Each Star Rating",
    color=rating_counts.index.astype(str),
    color_discrete_sequence=['#c54c82', '#ec729c', '#f4aeba' ,'#D3D3D3', '#fdfdcb']
    )

st.plotly_chart(fig_rating, width='stretch')

st.header("Genre-wise Analysis")

genre_avg = df.groupby('genre')['review/score'].mean().sort_values(ascending=False)

fig_genre = px.bar(
    x=genre_avg.values,
    y=genre_avg.index,
    orientation='h',
    labels={'x': 'Average Rating', 'y': 'Genre'},
    title="Average Rating by Genre",
    color=genre_avg.index,
    color_discrete_sequence=px.colors.qualitative.D3
)
st.plotly_chart(fig_genre, width='stretch')

st.subheader("Explore a Specific Genre")
selected_genre = st.selectbox("Choose a genre:", df['genre'].unique())

genre_df = df[df['genre'] == selected_genre]

col1, col2, col3 = st.columns(3)
col1.metric("Reviews in this genre", f"{len(genre_df):,}")
col2.metric("Average Rating", f"{genre_df['review/score'].mean():.2f}")
col3.metric("Average Sentiment", f"{genre_df['sentiment'].mean():.2f}")

st.header("Sentiment vs Rating Analysis")

sentiment_by_rating = df.groupby('review/score')['sentiment'].mean().reset_index()

fig_sentiment = px.bar(
    sentiment_by_rating,
    x='review/score',
    y='sentiment',
    labels={'review/score': 'Star Rating', 'sentiment': 'Average Sentiment Score'},
    title="Average Sentiment by Star Rating",
    color='sentiment',
    color_continuous_scale=['#e1eacd', '#01352c'] 
)
st.plotly_chart(fig_sentiment, width='stretch')

correlation = df['sentiment'].corr(df['review/score'])
st.info(f" Correlation between sentiment and rating: **{correlation:.3f}** — a moderate positive relationship, meaning sentiment alone doesn't fully explain star ratings.")

st.markdown("""
**Observation :** 1-star reviews show a near-neutral average sentiment (not strongly negative), 
suggesting that low ratings are often driven by factual complaints (pricing, content quality) 
rather than emotionally charged negative language.
""")

# ---- Topic Modeling Results   ----
st.header("Topic Modeling: What Do Reviews Talk About?")

st.markdown("Using NMF (Non-negative Matrix Factorization) on TF-IDF vectors to identify themes in reviews.")

tab1, tab2, tab3 = st.tabs(["Low Rating (1-2★)", "Mid Rating (3★)", "High Rating (4-5★)"])

with tab1:
    st.markdown("""
    - **Topic 1:** story, characters, novel, plot — readers unhappy with the story or characters
    - **Topic 2:** book, money, buy, pages — readers feel it wasn't worth the price
    - **Topic 3:** read, boring, reading, school — readers found it slow or boring
    - **Topic 4:** books, series, previous, plot — complaints about a specific book series
    """)

with tab2:
    st.markdown("""
    - **Topic 1:** novel, story, characters, life — mixed feelings about characters and story
    - **Topic 2:** book, good, information, interesting — liked the information but nothing special
    - **Topic 3:** read, just, like, really — general, mixed reading experience
    - **Topic 4:** series, books, characters, plot — mixed opinions about a book series
    """)

with tab3:
    st.markdown("""
    - **Topic 1:** story, characters, novel, love, family — readers felt emotionally connected to the characters
    - **Topic 2:** book, great, good, recommend — readers really liked it and would recommend it
    - **Topic 3:** series, science, fiction, trilogy — fans of the genre loved this series
    """)

st.caption("Note: Some topics reflect specific popular titles/authors (e.g., Foundation, 1984) rather than pure generic themes — a known limitation of word co-occurrence-based topic modeling, which could be refined further using Named Entity Recognition.")


# ---- Interactive Review Explorer  ----
st.header("Explore Individual Reviews")

col1, col2 = st.columns(2)

with col1:
    explore_genre = st.selectbox("Select Genre:", df['genre'].unique(), key="explore_genre")

with col2:
    explore_rating = st.selectbox("Select Rating:", sorted(df['review/score'].unique(), reverse=True), key="explore_rating")

filtered_reviews = df[
    (df['genre'] == explore_genre) & 
    (df['review/score'] == explore_rating)
]

st.write(f"Found **{len(filtered_reviews):,}** reviews matching this filter.")

if len(filtered_reviews) > 0:
    num_samples = min(5, len(filtered_reviews))
    sample_reviews = filtered_reviews.sample(num_samples, random_state=42)

    for idx, row in sample_reviews.iterrows():
        with st.expander(f"📖 {row['Title'][:60]} — Sentiment: {row['sentiment']:.2f}"):
            st.write(row['review/text'])
else:
    st.warning("No reviews found for this combination. Try a different genre or rating.")