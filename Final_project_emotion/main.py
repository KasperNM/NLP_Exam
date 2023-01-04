#Main
#Importing packages
import argparse
from data import read_data, preprocess
import pandas as pd
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig 
import numpy as np
from transformers import TFAutoModelForSequenceClassification
from scipy.special import softmax

def main(data="subset_for_emotion.csv"):
    #Reading in the data using a predefined function from data.py
    df = read_data(data)

    def sentiment_analysis(row):
        text = row['Line']  # Get the text from the 'Line' column
        text = preprocess(text)  # Preprocess the text
        encoded_input = tokenizer(text, return_tensors='pt')  # Encoding the text
        output = model(**encoded_input)  # Perform sentiment_emotion analysis
        scores = output[0][0].detach().numpy()  # Get the scores
        scores = softmax(scores)  # Apply softmax function to the scores
        ranking = np.argsort(scores)  # Reverse the rankings so that the highest score is first
        ranking = ranking[::-1]  # Reverse the rankings
        sentiment = config.id2label[ranking[0]]  # Get the label for the highest ranking
        return sentiment
    
    #Specifying the BERT emotion model
    MODEL = f"bhadresh-savani/distilbert-base-uncased-emotion"
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    config = AutoConfig.from_pretrained(MODEL)

    df['text'] = df['Line'].apply(preprocess)   # Apply the preprocess function to each row of the DataFrame
    # Encode the text data using the tokenizer
    text_data = df['Line'].values
    
    #Adding emotion as a column in the dataset
    df['Emotion_sentiment'] = df.apply(sentiment_analysis, axis=1)
    df.to_csv('emotion_df.csv', index=False)

def parseArguments():
    # Create argument parser
    parser = argparse.ArgumentParser()

    # Optional arguments
    parser.add_argument("-df", "--data", type=str, default='subset_for_emotion.csv')

    # Parse arguments
    args = parser.parse_args()

if __name__ == "__main__":
    args = parseArguments()
    main(args.data)

if __name__ == "__main__":
   #args = parseArguments()
    main()
