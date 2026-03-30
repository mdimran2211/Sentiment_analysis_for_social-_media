import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# --- Page Config ---
st.set_page_config(page_title="Sentiment Analyzer Pro", layout="wide")

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

# --- Load & Train Model (Cached for Speed) ---
@st.cache_resource
def train_model():
    df = pd.read_csv('sentimentdataset.csv')
    df['cleaned_text'] = df['Text'].apply(clean_text)
    df['Sentiment'] = df['Sentiment'].str.strip()
    
    tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
    X = tfidf.fit_transform(df['cleaned_text'])
    y = df['Sentiment']
    
    model = SVC(kernel='linear', probability=True)
    model.fit(X, y)
    return model, tfidf

model, tfidf = train_model()

# --- Streamlit UI ---
st.title("🚀 Social Media Sentiment Analysis Tool")
st.markdown("Developed with **SVM (Support Vector Machine)** as per Project Synopsis.")

# Sidebar for Navigation
option = st.sidebar.selectbox("Choose Action", ["Home & Manual Test", "Live Topic Analysis"])

if option == "Home & Manual Test":
    st.subheader("📝 Analyze Custom Text")
    user_input = st.text_area("Enter a social media post/comment:", placeholder="Type here...")
    
    if st.button("Predict Sentiment"):
        if user_input:
            cleaned = clean_text(user_input)
            vec = tfidf.transform([cleaned])
            prediction = model.predict(vec)[0]
            
            # Display Result with Color
            if "Positive" in prediction:
                st.success(f"Sentiment: {prediction} 😊")
            elif "Negative" in prediction:
                st.error(f"Sentiment: {prediction} 😠")
            else:
                st.info(f"Sentiment: {prediction} 😐")
        else:
            st.warning("Please enter some text first.")

elif option == "Live Topic Analysis":
    st.subheader("🌐 Real-time Simulation (API Fallback)")
    topic = st.text_input("Enter a trending topic:", "IPL 2024")
    
    if st.button("Fetch & Analyze"):
        # Simulated Live Tweets
        mock_tweets = [
            f"The atmosphere at the stadium for {topic} is electric!",
            f"I am really unhappy with the {topic} results today.",
            f"Just saw a post about {topic}, looks interesting.",
            f"Absolute disaster performance in {topic}. Boring!",
            f"Can't wait for the next update on {topic}! So excited."
        ]
        
        results_df = []
        for t in mock_tweets:
            cleaned = clean_text(t)
            vec = tfidf.transform([cleaned])
            pred = model.predict(vec)[0]
            results_df.append({"Tweet": t, "Sentiment": pred})
        
        st.table(pd.DataFrame(results_df))

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.write("✅ **Model:** SVM (Linear)")
st.sidebar.write("✅ **Dataset:** sentimentdataset.csv")
