import torch
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from transformers import AdamW
from torch.utils.data import DataLoader
from torch.nn import functional as F


# Detect GPU
if torch.cuda.is_available():
  device = torch.device("cuda")
  print('Using GPU ', torch.cuda.get_device_name(0)) 
else:
  device = torch.device("cpu")
  print('Using CPU')


# Load data
df = pd.read_csv('10K_text_price_label.csv')


# Split the dataset
doc_data = df[['Doc']].to_numpy()
doc_data = doc_data.reshape(doc_data.shape[0])
labels = df[['Label']].to_numpy()
labels = labels.reshape(labels.shape[0])

train_texts, test_texts, train_labels, test_labels = train_test_split(doc_data, labels, test_size=0.2, shuffle=True, random_state=0)


# Tokenize the text
from tokenizers import BertWordPieceTokenizer

# Here, we use the vocabulary from LoughranMcDonald corpus instead of the default one
vocab = 'voc_uniq.txt'
tokenizer = BertTokenizer(vocab)


# Train the data

# Set the parameters
EPOCHS = 20
BATCHES = 8
best_lr = 1e-5

# Turn labels and encodings into a Dataset object
class MyDataset(torch.utils.data.Dataset):
  def __init__(self, encodings, labels):
    self.encodings = encodings
    self.labels = labels

  def __getitem__(self, idx):
    item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    return item

  def __len__(self):
    return len(self.labels)

# Encoding the training data
train_encoding = tokenizer(list(train_texts), return_tensors='pt', padding=True, truncation=True, max_length=30)
# Turn into dataset object
train_dataset = MyDataset(train_encoding, train_labels)
# Use mini-bathces
train_loader = DataLoader(train_dataset, batch_size=BATCHES, shuffle=True)

# Encoding the testing data
test_encoding = tokenizer(list(test_texts), return_tensors='pt', padding=True, truncation=True, max_length=30)
test_dataset = MyDataset(test_encoding, test_labels)
test_loader = DataLoader(test_dataset, batch_size=BATCHES, shuffle=False)

# Bert model from Huggingface
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', return_dict=True, num_labels=3)

# Set the optimizer AdamW
optimizer = AdamW(model.parameters(), lr=best_lr)

# Implement early stopping
min_loss = float('inf')
epoch_count = 0
early_stop = False

# Put the model on device
model.to(device)

for epoch in range(EPOCHS):
  # Put the model in training mode
  model.train()

  train_loss = 0

  for batch in train_loader:
    optimizer.zero_grad()
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    batch_labels = batch['labels'].to(device)
    outputs = model(input_ids, attention_mask=attention_mask, labels=batch_labels)
    # Use cross entropy loss
    #loss = F.cross_entropy(outputs.logits, batch_labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    train_loss += loss.item()
  
  print("Current Epoch: {}".format(epoch + 1))
  print("------------------------------------------")
  print("Train loss: {}".format(train_loss))
  print()
  
  # Check whether to stop or not
  min_loss = min(train_loss, min_loss)
  if min_loss < train_loss:
    if epoch_count == 4:
      early_stop = True
      print("Stop training because of the early stop at epoch {}".format(epoch + 1))
      break
    else:
      epoch_count += 1
  else:
    # Reset the count
    epoch_count = 0


# Report the evaluation result on testing set
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

with torch.no_grad():
    total_loss = 0
    y_pred = None
    for batch in test_loader:
        
      input_ids = batch['input_ids'].to(device)
      attention_mask = batch['attention_mask'].to(device)
      batch_labels = batch['labels'].to(device)
      output = model(input_ids, attention_mask=attention_mask, labels=batch_labels)

      _, predicted_labels = torch.max(output.logits, 1)
      if y_pred is not None:
          y_pred = torch.cat((y_pred, predicted_labels), 0)
      else:
          y_pred = predicted_labels
 
precision, recall, f1, _ = precision_recall_fscore_support(test_labels, y_pred.cpu(), average='micro')
acc = accuracy_score(test_labels, y_pred.cpu())
print('Precison: {}'.format(precision))
print('Recall: {}'.format(recall))
print('F1 score: {}'.format(f1))
print('Accuracy: {}'.format(acc))
