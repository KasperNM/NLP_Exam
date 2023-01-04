import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

from tqdm import trange
import random

#Defining a function for reading in a dataframe for training and test
def read_data(train_path, test_path):
  train_df = pd.read_csv(train_path, encoding='latin-1', keep_default_na=False)
  test_df = pd.read_csv(test_path, encoding='latin-1' , keep_default_na=False)
  
  #Extracting the values from the line and gender columns in the trainingset
  train_text = train_df.Line.values
  train_labels = train_df.Gender.values

  #Extracting the values from the line and gender columns in the testset
  test_text = test_df.Line.values
  test_labels = test_df.Gender.values
  
  #Turns the values into tensors
  tensor_train_labels = torch.tensor(train_labels)
  tensor_test_labels = torch.tensor(test_labels)

  return train_text, train_labels, test_text, test_labels, tensor_train_labels, tensor_test_labels

#Function that returns a tokenizer that e.g. adds special tokens and returns attention mask
def preprocessing(input_text, tokenizer):
  return tokenizer.encode_plus(
                        input_text,
                        add_special_tokens = True,
                        max_length = 32,
                        pad_to_max_length = True,
                        return_attention_mask = True,
                        return_tensors = 'pt'
                   )

#Get the bert tokenizer
def preprocess_id_and_ams(train_text):
  bert_tokenizer = BertTokenizer.from_pretrained(
    'bert-base-uncased',
    do_lower_case = True
    )

  token_id = []
  attention_masks = []
  #Preprocess the data using both the bert_tokenizer and the predefined tokenizer
  for sample in train_text:
    encoding_dict = preprocessing(str(sample), bert_tokenizer)
    token_id.append(encoding_dict['input_ids']) 
    attention_masks.append(encoding_dict['attention_mask'])

  #appending values to token_id and attention_masks
  token_id = torch.cat(token_id, dim = 0)
  attention_masks = torch.cat(attention_masks, dim = 0)

  return token_id, attention_masks

#same step for test data
def preprocess_test(test_text):
  bert_tokenizer = BertTokenizer.from_pretrained(
    'bert-base-uncased',
    do_lower_case = True
    )
  test_token_id = []
  test_attention_masks = []

  for sample in test_text:
    test_encoding_dict = preprocessing(str(sample), bert_tokenizer)
    test_token_id.append(test_encoding_dict['input_ids']) 
    test_attention_masks.append(test_encoding_dict['attention_mask'])

  #same step for test data
  test_token_id = torch.cat(test_token_id, dim = 0)
  test_attention_masks = torch.cat(test_attention_masks, dim = 0)

  return test_token_id, test_attention_masks 

#Defining a function that splits the trainingdata and runs the TensorDataset and DataLoader functions
def get_validation_and_train_dataloader(tensor_train_labels, token_id, attention_masks, batch_size=64, val_ratio=0.025):
  
  # Splitting trainingset and validationset and then stratified by labels
  train_idx, val_idx = train_test_split(
      np.arange(len(tensor_train_labels)),
      test_size = val_ratio,
      shuffle = True,
      stratify = tensor_train_labels)

  # Train set
  train_set = TensorDataset(token_id[train_idx], 
                            attention_masks[train_idx], 
                            tensor_train_labels[train_idx])
  # Validation set
  val_set = TensorDataset(token_id[val_idx], 
                          attention_masks[val_idx], 
                          tensor_train_labels[val_idx])

  # Preparing the DataLoader function
  train_dataloader = DataLoader(
              train_set,
              sampler = RandomSampler(train_set),
              batch_size = batch_size
          )

  validation_dataloader = DataLoader(
              val_set,
              sampler = SequentialSampler(val_set),
              batch_size = batch_size
          )
  return train_dataloader, validation_dataloader
