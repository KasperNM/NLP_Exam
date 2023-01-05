#Main 
#Importing packages
import argparse
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import trange
import random
from data import read_data, preprocess_id_and_ams, get_validation_and_train_dataloader, preprocess_test

# Prepare performance metrics to evaluate training
def count_tp(predictions, labels):
  #Counting true positives
  return sum([predictions == labels and predictions == 1 for predictions, labels in zip(predictions, labels)])

def count_fp(predictions, labels):
  #Counting false Positives
  return sum([predictions != labels and predictions == 1 for predictions, labels in zip(predictions, labels)])

def count_tn(predictions, labels):
  #Counting true Negatives
  return sum([predictions == labels and predictions == 0 for predictions, labels in zip(predictions, labels)])

def count_fn(predictions, labels):
  #Counting false Negatives
  return sum([predictions != labels and predictions == 0 for predictions, labels in zip(predictions, labels)])

def b_metrics(predictions, labels):
  
  # Calculate accuracy, precision, recall, and specificity
  preds = np.argmax(predictions, axis = 1).flatten()
  labels = labels.flatten()
  # Counting true positives
  tp = count_tp(preds, labels)
  # Counting true negatives
  tn = count_tn(preds, labels)
  # Counting false positives
  fp = count_fp(preds, labels)
  # Counting true negatives
  fn = count_fn(preds, labels)
  # Calculating accuracy
  b_accuracy = (tp + tn) / len(labels)
  # Calculating precision
  b_precision = tp / (tp + fp) if (tp + fp) > 0 else 'nan'
  # Calculating recall
  b_recall = tp / (tp + fn) if (tp + fn) > 0 else 'nan'
  # Calculating specificity
  b_specificity = tn / (tn + fp) if (tn + fp) > 0 else 'nan'
  return b_accuracy, b_precision, b_recall, b_specificity

'''Main function: This function takes in other predefined functions from data.py to train
and evaluate the model on specified data'''
def main(traindata='train89.csv', testdata='test89.csv'):
  # Specifying the path to load in data for training and test data
  # Trainingsdata path
  train_path = './Data_for_classification/' + traindata
  # Testdata path
  test_path = './Data_for_classification/' + testdata
  
  # Reading in the data using a predefined function from data.py
  train_text, train_labels, test_text, test_labels, tensor_train_labels, tensor_test_labels = read_data(train_path, test_path)
  
  # Getting token ids and attention_masks from the preprocessed training text
  token_id, attention_masks = preprocess_id_and_ams(train_text)
  
  #Test set is preprocessed the same way
  test_token_id, test_attention_masks = preprocess_test(test_text)

  # Call the train and validation dataloader functions
  # The batch size is set to 64 and validation set i specified as 2.5% of trainingsdata
  train_dataloader, validation_dataloader = get_validation_and_train_dataloader(tensor_train_labels, token_id, attention_masks, 64, 0.025)

  # Loading in the BertForSequenceClassification model
  model = BertForSequenceClassification.from_pretrained(
      'bert-base-uncased',
      num_labels = 2,
      output_attentions = False,
      output_hidden_states = False,
  )

  # Learning rate is set to 2e-5 and a small epsilon to avoid division with zero
  optimizer = torch.optim.AdamW(model.parameters(), 
                                lr = 2e-5,
                                eps = 1e-08
                                )

  # This classification is run on cuda
  # If your machine is unable to run cuda the device is set to cpu

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  if (device == 'cuda'):
    model.cuda()
  else:
    pass

  # It is chosen to run training on 4 epochs
  n_epochs = 1

  for _ in trange(n_epochs, desc = 'Epoch'):
      
      #  Training 
      
      # The model is set to training mode
      model.train()
      
      # Tracking multiple variables
      # The training loss and the number of training examples and steps are initialized to 0
      tr_loss = 0
      nb_tr_examples, nb_tr_steps = 0, 0
      
      # A loop that iterates over trainingdata, in batches
      for step, batch in enumerate(train_dataloader):
          batch = tuple(t.to(device) for t in batch)
          b_input_ids, b_input_mask, b_labels = batch
          #For each batch of data, the optimizer's gradient is set to zero
          optimizer.zero_grad()
          # Forward pass
          train_output = model(b_input_ids, 
                              token_type_ids = None, 
                              attention_mask = b_input_mask, 
                              labels = b_labels)
          # training loss and gradients are computed
          train_output.loss.backward()
          #updating the models parameters
          optimizer.step()
          # Updating the variables we are tracking
          tr_loss += train_output.loss.item()
          nb_tr_examples += b_input_ids.size(0)
          nb_tr_steps += 1

      # Validation 

      # For evaluation on the validationdata the model is set to eval mode
      model.eval()

      # Tracking variables (these were defined earlier as well)
      val_accuracy = []
      val_precision = []
      val_recall = []
      val_specificity = []
      
      #iterate through batches
      for batch in validation_dataloader:
          batch = tuple(t.to(device) for t in batch)
          b_input_ids, b_input_mask, b_labels = batch
          with torch.no_grad():
            # Forward pass
            eval_output = model(b_input_ids, 
                                token_type_ids = None, 
                                attention_mask = b_input_mask)
          logits = eval_output.logits.detach().cpu().numpy()
          label_ids = b_labels.to('cpu').numpy()
          # Calculate validation metrics previously specified
          b_accuracy, b_precision, b_recall, b_specificity = b_metrics(logits, label_ids)
          # Getting the accuracy metric
          val_accuracy.append(b_accuracy)
          # We are only updating precision, recall, and specificity when the metric calculation is !=0
          # nan is ignored
          if b_precision != 'nan': val_precision.append(b_precision)
          
          if b_recall != 'nan': val_recall.append(b_recall)
          
          if b_specificity != 'nan': val_specificity.append(b_specificity)

      print('\n\t - Train loss: {:.4f}'.format(tr_loss / nb_tr_steps))
      print('\t - Validation Accuracy: {:.4f}'.format(sum(val_accuracy)/len(val_accuracy)))
      print('\t - Validation Precision: {:.4f}'.format(sum(val_precision)/len(val_precision)) if len(val_precision)>0 else '\t - Validation Precision: NaN')
      print('\t - Validation Recall: {:.4f}'.format(sum(val_recall)/len(val_recall)) if len(val_recall)>0 else '\t - Validation Recall: NaN')
      print('\t - Validation Specificity: {:.4f}\n'.format(sum(val_specificity)/len(val_specificity)) if len(val_specificity)>0 else '\t - Validation Specificity: NaN')

  eval_model(model, test_token_id, test_attention_masks, tensor_test_labels)

#Now we evaluate the trained model
def eval_model(model, test_token_id, test_attention_masks, tensor_test_labels, batch_size=64):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  if (device == 'cuda'):
    model.cuda()
  else:
    pass
  #Same preperation procedure for the testset 
  test_set = TensorDataset(test_token_id, test_attention_masks, tensor_test_labels)
  test_dataloader = DataLoader(test_set, batch_size=batch_size, sampler = RandomSampler(test_set), shuffle=False)

  # Initialize variables to store predictions and ground truth labels
  predictions = []
  true_labels = []

  # Set the model to evaluation mode
  model.eval()
  # Iterate over the test dataloader
  for inputs, attention_mask, labels in test_dataloader:
    # Move input and label tensors to the correct device
    inputs = inputs.to(device)
    labels = labels.to(device)
    attention_mask = attention_mask.to(device)  # Attention mask

    # Forward pass, make predictions
    with torch.no_grad():
      output = model(inputs, attention_mask=attention_mask)
      logits = output.logits
      pred = logits.argmax(dim=1)
      
    # Save predictions and true labels
    predictions.extend(pred)
    true_labels.extend(labels)
      
    # Save predictions and true labels
    predictions.extend(pred)
    true_labels.extend(labels)

  # Convert lists to tensors
  predictions = torch.tensor(predictions)
  true_labels = torch.tensor(true_labels)

  # Calculate accuracy
  accuracy = (predictions == true_labels).float().mean().item()
  print('Accuracy: ', accuracy)

  # Calculate precision, recall, and F1 score
  precision = precision_score(true_labels, predictions)
  recall = recall_score(true_labels, predictions)
  f1 = f1_score(true_labels, predictions)
  
  #Print performance metrics
  print('Precision: ', precision)
  print('Recall: ', recall)
  print('F1 score: ', f1)
  
  # Define the measures as a dictionary
  measures = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1}

  # Create a dataframe from the measures dictionary
  performance_metrics = pd.DataFrame.from_dict(measures, orient='index', columns=['Value'])

  # Export the dataframe to a CSV file
  performance_metrics.to_csv('./Results/model_performance.csv', index=True, header=True)

  # Calculate the confusion matrix
  num_classes = 2
  confusion_matrix = torch.zeros(num_classes, num_classes)
  for t, p in zip(true_labels, predictions):
    confusion_matrix[t, p] += 1
  print('Confusion matrix: \n', confusion_matrix)

  # Normalize the confusion matrix
  confusion_matrix = confusion_matrix.float() / confusion_matrix.sum(dim=1, keepdim=True)

  # Set up the plot
  fig, ax = plt.subplots()
  im = ax.imshow(confusion_matrix, cmap='Blues')
  class_names = ['Male', 'Female']

  # Set the tick marks and labels
  ax.set_xticks(np.arange(num_classes))
  ax.set_yticks(np.arange(num_classes))
  ax.set_xticklabels(class_names)
  ax.set_yticklabels(class_names)

  # Rotate the tick labels and set their alignment
  plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

  # Add the colorbar
  cbar = ax.figure.colorbar(im, ax=ax)

  # Add the labels and title
  ax.set_ylabel('True label')
  ax.set_xlabel('Predicted label')
  ax.set_title('Confusion matrix')

  # Show the plot
  plt.show()
  fig.savefig("./plots/confusion_matrix.png")

def parseArguments():
    # Create argument parser
    parser = argparse.ArgumentParser()

    # Optional arguments
    parser.add_argument("-tr", "--train", type=str, default='train89.csv')
    parser.add_argument("-te", "--test", type=str, default='test89.csv')

    # Parse arguments
    args = parser.parse_args()

    return args
#Run main()
if __name__ == "__main__":
    args = parseArguments()
    main(args.train, args.test)
