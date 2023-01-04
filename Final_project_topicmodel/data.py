import pandas as pd

#Defining a function to load in the data and extracting lines and gender_class
def read_data(df_path):
  data = pd.read_csv(df_path, encoding='latin-1', keep_default_na=False)
  lines = data["Line"].tolist()
  gender_class = data["Gender_label"].tolist()
  return data, lines, gender_class
