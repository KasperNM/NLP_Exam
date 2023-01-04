import pandas as pd

#Defining a function to read in the data
def read_data(df_path):
  df = pd.read_csv(df_path, encoding='latin-1', keep_default_na=False)
  
  return df

#preprocessing the text data
def preprocess(row):
    text = row  # Get the text from the current row
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)
