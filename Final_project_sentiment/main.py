#Main
#Importing packages
from data import read_data, preprocess
import pandas as pd
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig 
import numpy as np
from transformers import TFAutoModelForSequenceClassification
from scipy.special import softmax

#Main()
def main():
    #Reading in the data using a predefined function from data.py 
    df = read_data('Clean_scripts.csv')
    
    def sentiment_analysis(row):
        text = row['Line']  # Get the text from the 'Line' column
        text = preprocess(text)  # Preprocess the text
        encoded_input = tokenizer(text, return_tensors='pt')  # Encode the text
        output = model(**encoded_input)  # Perform sentiment analysis
        scores = output[0][0].detach().numpy()  # Get the scores
        scores = softmax(scores)  # Applying softmax function to the scores
        ranking = np.argsort(scores)  # Get the rankings
        ranking = ranking[::-1]  # Reverse the rankings so that the highest score is first
        sentiment = config.id2label[ranking[0]]  # Get the label for the highest ranking
        return sentiment
    
    #Specifying the BERT sentiment model
    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    config = AutoConfig.from_pretrained(MODEL)

    df['text'] = df['Line'].apply(preprocess)   # Apply the preprocess function to each row of the DataFrame
    # Encoding the text data using the tokenizer
    text_data = df['Line'].values
    
    #Adding the sentiment as a column in the dataset
    df['sentiment'] = df.apply(sentiment_analysis, axis=1)
    #Exporting as CSV
    df.to_csv('sentiment_neutral.csv', index=False)

if __name__ == "__main__":
   #args = parseArguments()
    main()
