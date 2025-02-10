import streamlit as st
import nltk
import spacy
import matplotlib.pyplot as plt
from transformers import pipeline
import random

# Load NLP models
nltk.download("vader_lexicon")
from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

nlp = spacy.load("en_core_web_sm")
emotion_pipeline = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

# Sample texts
sample_texts = [
    "The digital world is transforming the way we read and engage with text.",
    "Reading is an essential skill that shapes our understanding of the world.",
    "AI-driven education tools can personalize the learning experience for students."
]

# Streamlit UI
st.title("📖 AI-Powered Adaptive Reading Engagement")
st.write("Analyze how users engage with digital reading using AI-powered insights.")

# Text Input
text_option = st.selectbox("Choose a sample text or enter your own:", ["Use Sample"] + sample_texts)
if text_option == "Use Sample":
    text = st.text_area("Read this passage:", random.choice(sample_texts), height=150)
else:
    text = st.text_area("Enter your own text:", height=150)

# Sentiment Analysis
if st.button("Analyze Engagement"):
    if text:
        sentiment_score = sia.polarity_scores(text)
        emotion_results = emotion_pipeline(text)

        # Display Sentiment
        st.subheader("📊 Sentiment Analysis")
        st.write(f"Positive: {sentiment_score['pos'] * 100:.2f}%, Negative: {sentiment_score['neg'] * 100:.2f}%, Neutral: {sentiment_score['neu'] * 100:.2f}%")

        # Display Emotion
        st.subheader("🎭 Emotion Detection")
        top_emotion = max(emotion_results[0], key=lambda x: x['score'])
        st.write(f"Detected Emotion: **{top_emotion['label']}** (Confidence: {top_emotion['score']:.2f})")

        # Visualization
        labels = [e['label'] for e in emotion_results[0]]
        scores = [e['score'] for e in emotion_results[0]]
        fig, ax = plt.subplots()
        ax.bar(labels, scores)
        st.pyplot(fig)
    else:
        st.warning("Please enter a text to analyze.")
