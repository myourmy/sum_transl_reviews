# Sentiment Analysis and Summarization Tool
This tool performs sentiment analysis and text summarization on uploaded CSV files containing review data. It uses the streamlit, pandas, transformers, matplotlib, and googletrans libraries.

## Libraries
Here's an explanation of the libraries used in the code:
streamlit: Streamlit is a popular Python library used for building web applications for machine learning and data science projects. It simplifies the process of creating interactive web apps by allowing developers to write Python scripts that are automatically converted into web apps.
pandas: Pandas is a powerful data manipulation library in Python. It provides data structures like DataFrame and Series, which are ideal for working with structured data such as CSV files. Pandas is used here for reading, processing, and analyzing the uploaded CSV file.
transformers: Transformers is a library by Hugging Face that provides pre-trained models for natural language processing (NLP). It includes various models for tasks like text summarization, sentiment analysis, and translation. In this code, the library is used for text summarization and translation. 
matplotlib: Matplotlib is a popular data visualization library in Python. It provides a variety of functions for creating static, animated, and interactive plots. Matplotlib is used here for creating bar charts to visualize the number of reviews per restaurant and the average rating per restaurant.
googletrans: Googletrans is a Python wrapper around Google Translate API. It allows for easy translation of text between languages. In this code, it is used for translating reviews to a specified target language.

## Features
Upload CSV: Upload a CSV file containing review data.
Data Processing: Preprocesses the data by converting "Rating" to numeric and filling empty reviews with the mean rating value.
Sentiment Analysis: Analyzes the sentiment of reviews and categorizes them as positive, negative, or neutral.
Text Summarization: Summarizes the reviews to a specified maximum and minimum length.
Translation: Optionally translates reviews to a specified target language.
Visualization: Displays tables showing restaurant ratings and review counts, as well as unique reviews. Also provides visualizations of the number of reviews per restaurant and the average rating per restaurant.

##  Usage from User Side
Upload File: Click the "Choose a CSV file" button to upload a CSV file containing review data.



Data Processing: The tool preprocesses the data and displays two tables:
Table 1: Restaurant Ratings and Review Counts


Table 2: Unique Review with translation of Reviews for foreigners or other language speakers



Visualizations: Visualizations are displayed showing the number of reviews per restaurant and the average rating per restaurant.


##  Installation
To run this tool, you need to have Python installed. Install the required packages using pip:
pip install streamlit pandas transformers matplotlib googletrans==4.0.0-rc1

##  How to Run
Save the code in a Python file (e.g., sentiment_analysis_tool.py) and run it using Streamlit:
streamlit run sentiment_analysis_tool.py

##  Functions
load_models(): This function loads the sentiment analysis model and tokenizer required for text summarization. It returns these loaded models or returns None if there is an exception during loading.

This function creates a sentiment analysis pipeline using the Hugging Face pipeline function. 
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
Then creates a tokenizer for the BART model from Hugging Face’s transformers library and specifies the pretrained model. 
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn') 
Finally it loads a pretrained BART model as the tokenizer.
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn') 

summarize_text(text, max_length=130, min_length=30): This function takes a text input and uses the loaded tokenizer and model to generate a summary of the text. It returns the summarized text.

This function is used to encode initial text (inputs) and decode resulting text (return tokenizer.decode….) using tokens and other features such as number of beams (number of parallel sequences), length penalty etc.
preprocess_data(data): This function preprocesses the input data by converting the "Rating" column to numeric and filling empty reviews with the mean rating value. It returns the preprocessed data.

perform_sentiment_and_summarization(data): This function applies text summarization and sentiment analysis to all reviews in the input data. It adds columns for summary, sentiment, sentiment score, and weighted sentiment to the data and returns the modified data.

Inputs: The function takes DataFrame data as input, which is assumed to contain a column named "Review" containing text reviews.
Summarization: It applies the summarize_text function (assuming it's defined elsewhere) to each review in the "Review" column and stores the summarized text in a new column named "Summary".
Sentiment Analysis: It uses the sentiment_pipeline to perform sentiment analysis on the reviews. The pipeline is applied to the entire "Review" column, and the results are stored in a variable named sentiments.
Sentiment Mapping: It maps the sentiment labels ('POSITIVE', 'NEGATIVE', 'NEUTRAL') to numerical values (1, -1, 0) and stores the result in a new column named "Sentiment Score".
Weighted Sentiment: It calculates a weighted sentiment score by multiplying the "Rating" column (assumed to exist in data) with the "Sentiment Score" column. The result is stored in a new column named "Weighted Sentiment".
Return Preprocessed DataFrame: It returns the modified DataFrame data with additional columns for summary, sentiment score, and weighted sentiment.

translate_text(text, target_language): This function translates the input text to the specified target language using the Google Translate API. It returns the translated text.

This function uses the googletrans library, which provides a Python interface to the Google Translate API. It creates a Translator object, then uses the translate method to translate the input text (text) to the specified target language (target_language). The translated text is returned as a string.
generate_tables(data, target_language='Spanish'): This function generates two tables from the input data:
Table 1: Aggregates average rating and number of reviews per restaurant.
Table 2: Contains unique reviews, optionally translated to the target language. It returns both tables.

##  Customization Options
 Target Language: You can change the target language for translation by modifying the `target_language` parameter in the `generate_tables` function. For example, to translate reviews to French, set `target_language='French'`.
Preprocessing Steps: Modify the `preprocess_data` function to customize how the data is preprocessed. For example, you can add additional cleaning steps or change how missing values are handled.
Number of rows: The tool processes 10.000 rows of the uploaded CSV file for demonstration purposes. You can modify this limit as needed.
Testing and Validation
The tool has been tested with various CSV files containing review data to ensure that it accurately performs sentiment analysis, text summarization, and translation. Additionally, manual validation has been performed on the tool's output to verify the correctness of the results.
Contributing Guidelines
Contributions to the sentiment analysis and summarization tool are welcome! If you'd like to contribute, please check the existing documantation:

https://github.com/myourmy/sum_transl_reviews/tree/main 


