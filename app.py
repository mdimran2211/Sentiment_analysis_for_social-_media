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
    # Make sure sentimentdataset.csv is in your GitHub folder
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

# --- Streamlit UI Header ---
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
            if any(word in prediction for word in ["Positive", "Joy", "Happy"]):
                st.success(f"Classification: {prediction} 😊")
            elif any(word in prediction for word in ["Negative", "Angry", "Sad"]):
                st.error(f"Classification: {prediction} 😠")
            else:
                st.info(f"Classification: {prediction} 😐")
        else:
            st.warning("Please enter some text first.")

elif option == "Live Topic Analysis":
    st.subheader("🌐 Real-time Simulation (API Fallback)")
    topic = st.text_input("Enter a trending topic to simulate analysis:", "snapchat")
    
    if st.button("Fetch & Analyze"):
        # Wahi exact sentences jo aapke screenshot mein hain
        mock_tweets = [
            f"The future of {topic} looks incredibly promising and bright!",
            f"I am really concerned about the impact of {topic} on jobs.",
            f"Just saw a new update about {topic}, it's quite revolutionary.",
            f"Absolute disaster implementation of {topic}. Very disappointed.",
            f"Can't wait to see how {topic} evolves this year! High hopes."
        ]
        
        results_df = []
        for t in mock_tweets:
            cleaned = clean_text(t)
            vec = tfidf.transform([cleaned])
            pred = model.predict(vec)[0]
            results_df.append({"Tweet": t, "Sentiment": pred})
        
        # Display as Table like in your screenshot
        st.table(pd.DataFrame(results_df))

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.write("✅ **Model:** SVM (Linear)")
st.sidebar.write("✅ **Dataset:** sentimentdataset.csv")
