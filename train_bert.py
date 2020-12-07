import torch
import sys
import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from transformers import AdamW
from torch.utils.data import DataLoader
from torch.nn import functional as F

from utils.dataset import MyDataset
from config import CONFIG_BERT

cfg = CONFIG_BERT()

if torch.cuda.is_available():
    device = torch.device("cuda")
    print('Using GPU ', torch.cuda.get_device_name(0)) 
else:
    device = torch.device("cpu")
    print('Using CPU')
    
    
dataset = sys.argv[1]
if len(sys.argv) != 2:
    sys.exit("Use: python train_bert.py <dataset>")
cfg.dataset = dataset    


df_train = pd.read_csv('cleaned_data/' + cfg.dataset + '/train.csv')
df_test = pd.read_csv('cleaned_data/' + cfg.dataset + '/test.csv')

print(df_train.head())


train_texts, train_labels = df_train['title'].to_numpy(), df_train['label'].to_numpy()
test_texts, test_labels = df_test['title'].to_numpy(), df_test['label'].to_numpy()


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_encodings = tokenizer(list(train_texts), return_tensors='pt', truncation=True, padding=True, max_length=cfg.max_length)

train_dataset = MyDataset(train_encodings, train_labels)
train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)


model = BertForSequenceClassification.from_pretrained('bert-base-uncased', return_dict=True, num_labels=cfg.num_labels)
model.to(device)



optim = AdamW(model.parameters(), lr=cfg.learning_rate)

for epoch in range(cfg.epochs):
   
    model.train()
    train_loss = 0
    for batch in train_loader:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        batch_labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=batch_labels)
        loss = F.cross_entropy(outputs.logits, batch_labels)
        loss.backward()
        optim.step()

        train_loss += loss.item()
    
    print("Epoch {}/{}".format(epoch+1, cfg.epochs))
    print("-"*15)
    print("Train loss: {}".format(train_loss))
    
    
    
    
test_encodings = tokenizer(list(test_texts), return_tensors='pt', truncation=True, padding=True, max_length=cfg.max_length)
test_dataset = MyDataset(test_encodings, test_labels)

test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

model.eval()
all_preds = None

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        batch_labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=batch_labels)

        _, preds = torch.max(outputs.logits, dim=1)
        all_preds = preds if all_preds == None else torch.cat((all_preds, preds), 0)


precision, recall, f1, _ = precision_recall_fscore_support(test_labels, all_preds.cpu(), average='micro')
accuracy = accuracy_score(test_labels, all_preds.cpu())

print('Accuracy : {}'.format(accuracy))
print('Precison : {}'.format(precision))
print('Recall   : {}'.format(recall))
print('F1 score : {}'.format(f1))