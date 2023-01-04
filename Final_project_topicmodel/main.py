#Main
#Importing packages
import transformers
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from data import read_data

#Main
def main():
    #Loading in the data and extracting lines and gender_class
    data, lines, gender_class = read_data('Clean_scripts.csv')
    
    #Removing stopwords
    vectorizer_model = CountVectorizer(stop_words="english")
    ctfidf_model = ClassTfidfTransformer()
    #Defining the model
    '''vectorizer_model: removes stopwords (after embedding).
    nr_topics: is for topic reduction.
    n_gram_range: decides how many tokens can be counted as 1 term (e.g. "New" "York")'''
    topic_model = BERTopic(vectorizer_model = vectorizer_model, 
                       nr_topics = 'auto',
                       n_gram_range = (1,2),
                       min_topic_size = 100,
                       ctfidf_model=ctfidf_model)
    #Extracting topics into a dataframe
    topics, probs = topic_model.fit_transform(lines)
    topic_model.get_topic_info()
    topic_model.visualize_barchart(top_n_topics=200)
    topics_per_class = topic_model.topics_per_class(lines, classes=gender_class)
    topics_per_class.to_csv("topic.csv")
