#Main
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

def b_tp(preds, labels):
  '''Returns True Positives (TP): count of correct predictions of actual class 1'''
  return sum([preds == labels and preds == 1 for preds, labels in zip(preds, labels)])

def b_fp(preds, labels):
  '''Returns False Positives (FP): count of wrong predictions of actual class 1'''
  return sum([preds != labels and preds == 1 for preds, labels in zip(preds, labels)])

def b_tn(preds, labels):
  '''Returns True Negatives (TN): count of correct predictions of actual class 0'''
  return sum([preds == labels and preds == 0 for preds, labels in zip(preds, labels)])

def b_fn(preds, labels):
  '''Returns False Negatives (FN): count of wrong predictions of actual class 0'''
  return sum([preds != labels and preds == 0 for preds, labels in zip(preds, labels)])

def b_metrics(preds, labels):
  '''
  Returns the following metrics:
    - accuracy    = (TP + TN) / N
    - precision   = TP / (TP + FP)
    - recall      = TP / (TP + FN)
    - specificity = TN / (TN + FP)
  '''
  preds = np.argmax(preds, axis = 1).flatten()
  labels = labels.flatten()
  tp = b_tp(preds, labels)
  tn = b_tn(preds, labels)
  fp = b_fp(preds, labels)
  fn = b_fn(preds, labels)
  b_accuracy = (tp + tn) / len(labels)
  b_precision = tp / (tp + fp) if (tp + fp) > 0 else 'nan'
  b_recall = tp / (tp + fn) if (tp + fn) > 0 else 'nan'
  b_specificity = tn / (tn + fp) if (tn + fp) > 0 else 'nan'
  return b_accuracy, b_precision, b_recall, b_specificity

def main():
  train_path = '/NLP_Exam/Final_project_classifier/Data_for_classification/train89.csv'
  test_path = '/NLP_Exam/Final_project_classifier/testset_for_classifier/Data_for_classification/test89.csv'

  train_text, train_labels, test_text, test_labels, tensor_train_labels, tensor_test_labels = read_data(train_path, test_path)

  token_id, attention_masks = preprocess_id_and_ams(train_text)

  train_dataloader, validation_dataloader = get_validation_and_train_dataloader(tensor_train_labels, token_id, attention_masks, 64, 0.025)

  test_token_id, test_attention_masks = preprocess_test(test_text)
  # Load the BertForSequenceClassification model
  model = BertForSequenceClassification.from_pretrained(
      'bert-base-uncased',
      num_labels = 2,
      output_attentions = False,
      output_hidden_states = False,
  )

  # Recommended learning rates (Adam): 5e-5, 3e-5, 2e-5. See: https://arxiv.org/pdf/1810.04805.pdf
  optimizer = torch.optim.AdamW(model.parameters(), 
                                lr = 2e-5,
                                eps = 1e-08
                                )

  # Run on GPU preferable:

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  if (device == 'cuda'):
    model.cuda()
  else:
    pass

  print(device)
  
  # Recommended number of epochs: 2, 3, 4. See: https://arxiv.org/pdf/1810.04805.pdf
  epochs = 4

  for _ in trange(epochs, desc = 'Epoch'):
      
      # ========== Training ==========
      
      # Set model to training mode
      model.train()
      
      # Tracking variables
      tr_loss = 0
      nb_tr_examples, nb_tr_steps = 0, 0

      for step, batch in enumerate(train_dataloader):
          batch = tuple(t.to(device) for t in batch)
          b_input_ids, b_input_mask, b_labels = batch
          optimizer.zero_grad()
          # Forward pass
          train_output = model(b_input_ids, 
                              token_type_ids = None, 
                              attention_mask = b_input_mask, 
                              labels = b_labels)
          # Backward pass
          train_output.loss.backward()
          optimizer.step()
          # Update tracking variables
          tr_loss += train_output.loss.item()
          nb_tr_examples += b_input_ids.size(0)
          nb_tr_steps += 1

      # ========== Validation ==========

      # Set model to evaluation mode
      model.eval()

      # Tracking variables 
      val_accuracy = []
      val_precision = []
      val_recall = []
      val_specificity = []

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
          # Calculate validation metrics
          b_accuracy, b_precision, b_recall, b_specificity = b_metrics(logits, label_ids)
          val_accuracy.append(b_accuracy)
          # Update precision only when (tp + fp) !=0; ignore nan
          if b_precision != 'nan': val_precision.append(b_precision)
          # Update recall only when (tp + fn) !=0; ignore nan
          if b_recall != 'nan': val_recall.append(b_recall)
          # Update specificity only when (tn + fp) !=0; ignore nan
          if b_specificity != 'nan': val_specificity.append(b_specificity)

      print('\n\t - Train loss: {:.4f}'.format(tr_loss / nb_tr_steps))
      print('\t - Validation Accuracy: {:.4f}'.format(sum(val_accuracy)/len(val_accuracy)))
      print('\t - Validation Precision: {:.4f}'.format(sum(val_precision)/len(val_precision)) if len(val_precision)>0 else '\t - Validation Precision: NaN')
      print('\t - Validation Recall: {:.4f}'.format(sum(val_recall)/len(val_recall)) if len(val_recall)>0 else '\t - Validation Recall: NaN')
      print('\t - Validation Specificity: {:.4f}\n'.format(sum(val_specificity)/len(val_specificity)) if len(val_specificity)>0 else '\t - Validation Specificity: NaN')

  eval_model(model, test_token_id, test_attention_masks, tensor_test_labels)


def eval_model(model, test_token_id, test_attention_masks, tensor_test_labels, batch_size=64):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  if (device == 'cuda'):
    model.cuda()
  else:
    pass
  # Indices of the train and validation splits stratified by labels
  test_set = TensorDataset(test_token_id, test_attention_masks, tensor_test_labels)
  test_dataloader = DataLoader(test_set, batch_size=batch_size, sampler = RandomSampler(test_set), shuffle=False)

  # Initialize variables to store predictions and ground truth labels
  predictions = []
  true_labels = []

  # Set the model to evaluation mode
  model.eval()
  # Iterate over the test dataloader
  # Iterate over the test dataloader
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

  print('Precision: ', precision)
  print('Recall: ', recall)
  print('F1 score: ', f1)

  # Calculate the confusion matrix
  num_classes = 2
  confusion_matrix = torch.zeros(num_classes, num_classes)
  for t, p in zip(true_labels, predictions):
    confusion_matrix[t, p] += 1
  print('Confusion matrix: \n', confusion_matrix)

  import matplotlib.pyplot as plt

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
  fig.savefig("/NLP_Exam/Final_project_classifier/plots/confusion_matrix.png")


if __name__ == "__main__":
   #args = parseArguments()
    main()