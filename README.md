# Movie-Review-classifier-

This project analyzes the sentiment of film reviews from the IMDB dataset. Using recurrent neural networks (RNNs) with LSTM layers, the model classifies reviews as positive or negative. The project involves data preprocessing and training a binary classification model to accurately predict the sentiment of new reviews.

The data set can be found on Kaggle https://www.kaggle.com/datasets/vishakhdapat/imdb-movie-reviews. 

Step 1 - Data Preprocessing for Movie Review Sentiment Analysis

Objective:
The goal is to preprocess the IMDb movie review dataset to enable accurate training and testing of a sentiment analysis model. Our target is to classify reviews as either positive or negative using natural language processing techniques.

Steps Taken:

    Data Import:
        The IMDb movie review dataset was imported from a CSV file containing two columns: review and sentiment.

    Setup NLTK Resources:
        The necessary stopwords and wordnet datasets were downloaded via NLTK to facilitate text cleaning.

    Text Cleaning Process:
        A function named clean_review was developed to handle the following tasks:
            Remove HTML Tags: Regular expressions stripped out any HTML markup present in the reviews.
            Convert to Lowercase: All text was converted to lowercase to standardize the data.
            Remove Punctuation and Digits: A regular expression filtered out punctuation marks and numbers.
            Remove Stopwords: Stopwords like "the" and "is" were filtered out using NLTK's stopwords corpus.
            Lemmatization: Words were reduced to their base forms using the WordNet lemmatizer, improving normalization.

    Apply Cleaning to Dataset:
        The clean_review function was applied to each review in the dataset to generate a new clean_review column containing cleaned text.

    Label Mapping:
        The sentiment column was mapped to numerical values: 1 for positive, 0 for negative.

    Data Splitting:
        The dataset was split into training and testing sets in an 80-20 ratio.
        The training set contains 40,000 samples, while the testing set has 10,000 samples.

    Exporting Results:
        The preprocessed training and testing sets were saved to new CSV files for future model training and evaluation.


