import streamlit as st
import pandas as pd
import joblib
import re
import nltk
import plotly.express as px
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

df = pd.read_csv('cleaned_sentiment_dataset.csv')

# Load model and vector
model = joblib.load('svm_model.pkl')
vector = joblib.load('tfidf_vectorizer.pkl')

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

# Streamlit app
st.title('Sentiment Analysis Web App')

st.write("This app uses a Support Vector Machine (SVM) model to classify the sentiment of your comments as positive, neutral, or negative.")

text_input = st.text_area("Enter your comment here:", height=100)
if text_input:
    cleaned_text = preprocess_text(text_input)
    vectorized_text = vector.transform([cleaned_text])
    prediction = model.predict(vectorized_text)
    sentiment = prediction[0]
    
    if sentiment == "positive":
        color = 'green'
    elif sentiment == "neutral":
        color = 'yellow'
    else:
        color = 'red'
        
    st.markdown(
        f"#### Sentiment: <span style='color:{color}; font-weight:bold;'>{sentiment.capitalize()}</span>",
        unsafe_allow_html=True
    )
else:
    st.markdown("#### Sentiment: -")
    
st.subheader("📊 Dataset Insights")

col1, col2= st.columns(2)

with col1:
    st.subheader("Platform Distribution")
    fig_platform = px.pie(df, names='Platform', title='Platform Distribution')
    st.plotly_chart(fig_platform, use_container_width=True)

with col2:
    st.subheader("Sentiment Distribution")
    fig_sentiment = px.pie(df, names='sentiment', title='Sentiment Distribution')
    st.plotly_chart(fig_sentiment, use_container_width=True)

platforms = df['Platform'].unique()
selected_platform = st.selectbox("Select a platform to filter sentiment distribution:", platforms)

filtered_df = df[df['Platform'] == selected_platform]

st.subheader(f"Sentiment Distribution for {selected_platform}")
fig_sentiment = px.pie(filtered_df, names='sentiment', title=f'Sentiment on {selected_platform}')
st.plotly_chart(fig_sentiment, use_container_width=True)