# NLP_Exam project
This project includes 5 folders:
## Preprocessing and analysis scripts and data
Includes a folder with raw script data, a folder with preprocessed training and test data for the classification analysis from the 'Preprocessing_for_classification.Rmd',
a folder with datasets from the sentiment analysis, and a folder with the cleaned dataframe after the initial preprocessing done in 'Data_cleaning_and_preprocessing.Rmd'

## Final_project_classifier
Includes the full pipeline for the binary classification analysis using BERT, which is used in our project. It consists of a main.py and data.py file. 
To run the full analysis run the following code in the terminal:
```
bash run.sh
```
To specify which dataset you want to use for the training and testing all the datasets are stored in the folder 'Data_for_classification'
the run.sh file looks like this:
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
The analysis will output a CSV.file in the 'Results' folder with performance metrics and attach a confusion matrix within the folder.

## Final_project_sentiment
Includes a full pipeline for sentiment analyses done on the data in our project. It consists of a main.py and a data.py file. 
To run the full analysis the same code as above can be run:
```
bash run.sh
```
The default dataset is a subset, which makes it easier to test the analysis without spending too much time
to run the analysis on the full data set type:
```
#!/usr/bin/bash

python3 -m venv env

source ./env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

python3 main.py --data Clean_scripts.csv

deactivate 
```

