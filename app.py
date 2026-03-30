import streamlit as st
import pandas as pd
import re
import nltk
import random
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# --- Page Config ---
st.set_page_config(page_title="Sentiment Analyzer Pro", layout="wide", page_icon="🚀")

# --- NLP Resources ---
@st.cache_resource
def load_resources():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()
    return stop_words, ps

stop_words, ps = load_resources()

def clean_text(text):
    text = re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", str(text))
    text = text.lower()
    words = text.split()
    cleaned = [ps.stem(w) for w in words if w not in stop_words]
    return " ".join(cleaned)

# --- Load & Train Model ---
@st.cache_resource
def train_model():
    df = pd.read_csv('sentimentdataset.csv')
    df['cleaned_text'] = df['Text'].apply(clean_text)
    df['Sentiment'] = df['Sentiment'].str.strip()
    
    # Balanced weights taaki model sirf positive na seekhe
    tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
    X = tfidf.fit_transform(df['cleaned_text'])
    y = df['Sentiment']
    
    model = SVC(kernel='linear', probability=True, class_weight='balanced')
    model.fit(X, y)
    return model, tfidf

model, tfidf = train_model()

# --- UI Header (Exactly like your screenshot) ---
st.title("🚀 Automated Sentiment Analysis Tool")
st.markdown("### Harnessing the power of SVM (Support Vector Machine) for real-time classification of social media text and trends.")
st.info("This system uses machine learning to categorize sentiments from unstructured social media data.")

# --- Sidebar ---
st.sidebar.header("Navigation")
option = st.sidebar.selectbox("Choose Action", ["Live Topic Analysis", "Home & Manual Test"])

st.sidebar.markdown("---")
st.sidebar.subheader("Project Details")
st.sidebar.write("✅ **Model Architecture:** SVM (Linear Kernel)")
st.sidebar.write("✅ **Feature Extraction:** TF-IDF Vectorizer")
st.sidebar.write("✅ **Source Dataset:** sentimentdataset.csv")

if option == "Live Topic Analysis":
    st.subheader("🌐 Real-time Simulation (Market Trend Analysis)")
    topic = st.text_input("Enter a trending topic to simulate analysis:", "Artificial Intelligence")
    
    if st.button("Fetch & Analyze Trends"):
        # Yahan humne diverse templates dale hain taaki results mix aayein
        templates = [
            f"The future of {topic} looks incredibly promising and bright!", # Positive
            f"I am really concerned about the negative impact of {topic} on jobs.", # Negative
            f"Just saw a new technical update about {topic}, it is quite complex.", # Neutral
            f"Absolute disaster implementation of {topic}. Very disappointed.", # Negative
            f"Is anyone else using {topic} today? Just wondering.", # Neutral
            f"I love how {topic} makes my work so much easier!", # Positive
            f"This new version of {topic} is full of bugs and very slow.", # Negative
            f"Comparing {topic} with other alternatives in the market." # Neutral
        ]
        
        # Har baar random 5 uthayega
        selected_posts = random.sample(templates, 5)
        
        results_df = []
        for t in selected_posts:
            cleaned = clean_text(t)
            vec = tfidf.transform([cleaned])
            pred = model.predict(vec)[0]
            results_df.append({"Social Media Post": t, "Sentiment Prediction": pred})
        
        st.table(pd.DataFrame(results_df))

elif option == "Home & Manual Test":
    st.subheader("📝 Analyze Custom Text")
    user_input = st.text_area("Enter a social media post/comment:", placeholder="Type here...")
    if st.button("Predict Sentiment"):
        if user_input:
            cleaned = clean_text(user_input)
            vec = tfidf.transform([cleaned])
            prediction = model.predict(vec)[0]
            st.write(f"### Predicted Sentiment: {prediction}")
