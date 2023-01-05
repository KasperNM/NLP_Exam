# NLP_Exam project
This project includes five folders:
The four "Final_project..." folders are analysis pipelines constructed by .py files in Python3. 
## Preprocessing and analysis scripts and data
R files created in Rstudio
Includes a folder with raw script data, a folder with preprocessed training and test data for the classification analysis from the 'Preprocessing_for_classification.Rmd',
a folder with datasets from the sentiment analysis, and a folder with the cleaned dataframe after the initial preprocessing done in 'Data_cleaning_and_preprocessing.Rmd' 

## Final_project_classifier
Includes the full pipeline for the binary classification analysis using BERT, which is used in our project. 
To run the analysis on the default set, run the following code in the terminal:
```
bash run.sh
```
To specify which dataset you want to use for the training and testing all the datasets are stored in the folder 'Data_for_classification'.
The run.sh file looks like this:
```
#!/usr/bin/bash

python3 -m venv env

source ./env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

python3 main.py 

deactivate
```
To select a given training and testing set type in the following within the run.sh file after 'python3 main.py':
```
#!/usr/bin/bash

python3 -m venv env

source ./env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

python3 main.py --train train94.csv --test test94.csv

deactivate
```
This example trains the model on train data from the years 1994-1997 and testing it data from 1994-1997. 
The default data sets are the data from 1989-1993
The analysis will output a CSV.file called 'model_performance.csv' with performance metrics and attach a confusion matrix within the folder.

## Final_project_sentiment
Includes the full pipeline for sentiment analyses done on the data in our project. 
To run the full analysis the same code as above can be run in the terminal:
```
bash run.sh
```
The default dataset is a subset, which makes it easier to test the analysis without spending too much time
to run the analysis on the full data set type the following in the 'run.sh' file before running 'bash run.sh' from the terminal:
```
#!/usr/bin/bash

python3 -m venv env

source ./env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

python3 main.py --data Clean_scripts.csv

deactivate 
```
The analysis outputs a new dataframe with a column called 'sentiment' with the highest scoring sentiment of that line

## Final_project_emotion
Includes the full pipeline for multi-label sentiment analysis of emotions done in our project. 
To run the code on a subset of data, type the following in the terminal:
```
bash run.sh
```
To run the analysis on the full dataset edit the run.sh file so it looks like this and run 'bash run.sh' from the terminal again:
```
#!/usr/bin/bash

python3 -m venv env

source ./env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

python3 main.py --data neutral_removed.csv

deactivate 
```
The model outputs a new dataframe with the column emotions added with the highest scoring emotion for each line.

## Final_project_topicmodel
Includes the full pipeline for using BERT for topic modelling on the scriptdata, which was done in our project.
To run on a subset of data run the following in the terminal:
```
bash run.sh
```
To run the analysis on the full dataset edit the 'run.sh' file like this before running 'bash run.sh' from the terminal:
```
#!/usr/bin/bash

python3 -m venv env

source ./env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

python3 main.py --data subset_for_topic.csv

deactivate 
```
This model outputs a dataframe with topics, frequency, and gender
Note: The topic model has occasionally shown to have some trouble running in VScode due to the hdbscan package from bertopic. However, The analysis ran smoothly in google colab
