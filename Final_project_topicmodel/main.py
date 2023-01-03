import transformers
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from data import read_data

def main():
    data, lines, gender_class = read_data('/NLP_Exam/Final_project_topic/Clean_scripts.csv')
    
    vectorizer_model = CountVectorizer(stop_words="english")
    ctfidf_model = ClassTfidfTransformer()

    topic_model = BERTopic(vectorizer_model = vectorizer_model, 
                       nr_topics = 'auto',
                       n_gram_range = (1,2),
                       min_topic_size = 100,
                       ctfidf_model=ctfidf_model)
    topics, probs = topic_model.fit_transform(lines)
    topic_model.get_topic_info()
    topic_model.visualize_barchart(top_n_topics=200)
    topics_per_class = topic_model.topics_per_class(lines, classes=gender_class)
    topics_per_class.to_csv("/NLP_Exam/Final_project_topic/Final_project_topic/topic.csv")