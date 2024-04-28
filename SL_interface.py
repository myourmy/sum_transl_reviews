import streamlit as st
import pandas as pd
from transformers import pipeline, BartTokenizer, BartForConditionalGeneration
import matplotlib.pyplot as plt
from googletrans import Translator
@st.cache_resource
def load_models():
    """"this function is used to load sentiment analysis pre-trained model and tokenizer from Hugging Face
    transformers libraries and use them in the functions below"""
    try:
        # creating sentiment analysis pipleine based on HF pipleine
        sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
        model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
        return sentiment_pipeline, tokenizer, model
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        return None, None, None

sentiment_pipeline, tokenizer, model = load_models()
# stop if at least of them return None
if sentiment_pipeline is None or tokenizer is None or model is None:
    st.stop()

st.title("Sentiment Analysis and Summarization Tool")


def summarize_text(text, max_length=130, min_length=30):
    """function used to summarize text"""
    inputs = tokenizer.encode("summarize: " + str(text), return_tensors="pt", max_length=1024, truncation=True)
    # convert text to tokens, "return_tensors="pt"- will return tokens as PyTorch tensors
    summary_ids = model.generate(inputs, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4,
                                 early_stopping=True)
    # num beams - 4 sequences in parallel and select the best
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    # decoder returns tokens back  to text

def preprocess_data(data):
    """function to convert "Rating" to numeric + to fill empty reviews by mean_Rating value """
    data.loc[:, 'Rating'] = pd.to_numeric(data['Rating'], errors='coerce')
    # coerce = if any value cannot be converted it's = NaN
    mean_rating = data['Rating'].mean()
    # fill NaN by mean Rating
    data.loc[:, 'Rating'] = data['Rating'].fillna(mean_rating)
    return data

def perform_sentiment_and_summarization(data):
    """function used to apply summarized review to all the reviews"""
    data['Summary'] = data['Review'].apply(summarize_text) # invoke summarize_ text function
    # sentiment analysis on the review using pipleine and store results in the 'sentiments'
    sentiments = sentiment_pipeline(data['Review'].tolist(), truncation=True, max_length=512)
    # sentiment mapping - (-1, 0, 1)
    data['Sentiment'] = [sent['label'] for sent in sentiments]
    "mapping rates to 3 categories"
    data['Sentiment Score'] = data['Sentiment'].map({'POSITIVE': 1, 'NEGATIVE': -1, 'NEUTRAL': 0}).astype(int)
    data['Weighted Sentiment'] = data['Sentiment Score'] * data['Rating']
    return data

def translate_text(text, target_language):
    """Translate text to the specified target language using Google Translate API."""
    translator = Translator()
    translated_text = translator.translate(text, dest=target_language)
    return translated_text.text

def generate_tables(data, target_language= 'German'):
    table1 = data.groupby('Restaurant').agg(Average_Rating=('Rating', 'mean'),
                                            Number_of_Reviews=('Restaurant', 'size')).reset_index()
    table2 = data.drop_duplicates(subset=['Reviewer']).reset_index(drop=True)

    if target_language:
        table2['Translated_Review'] = table2['Review'].apply(lambda x: translate_text(x, target_language))
    else:
        table2['Translated_Review'] = table2['Review']

    return table1, table2


uploaded_file = st.file_uploader("Choose a CSV file")
if uploaded_file is not None:
    with st.spinner('Loading and processing data...'):
        data = pd.read_csv(uploaded_file)
        data = preprocess_data(data.head(750)) #invoke preprocess_data function
        data = perform_sentiment_and_summarization(data.head(750)) # invoke perform_sentiment_and_summarization function

    table1, table2 = generate_tables(data)

    # Displaying Tables
    st.write("Table 1: Restaurant Ratings and Review Counts")
    st.dataframe(table1)

    st.write("Table 2: Unique Reviews")
    st.dataframe(table2)

    # Visualizations
    st.subheader("Number of Reviews per Restaurant")
    fig, ax = plt.subplots()
    table1.plot(kind='bar', x='Restaurant', y='Number_of_Reviews', ax=ax, color='skyblue', legend=None)
    plt.xlabel('Restaurant')
    plt.ylabel('Number of Reviews')
    st.pyplot(fig)

    st.subheader("Average Rating per Restaurant")
    fig, ax = plt.subplots()
    table1.plot(kind='bar', x='Restaurant', y='Average_Rating', ax=ax, color='lightgreen', legend=None)
    plt.xlabel('Restaurant')
    plt.ylabel('Average Rating')
    st.pyplot(fig)
else:
    st.write("Please upload a file to proceed.")
