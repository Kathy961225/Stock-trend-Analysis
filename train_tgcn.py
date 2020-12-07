import numpy as np


import time
import tensorflow as tf
from tensorflow.keras import optimizers
from sklearn import metrics

from utils.utils import *
from models.gcn import GCN
from models.mlp import MLP

from config import CONFIG_TGCN
import os


cfg = CONFIG_TGCN()

# set random seed
seed = 6606
np.random.seed(seed)
tf.random.set_seed(seed)


dataset = sys.argv[1]
if len(sys.argv) != 2:
    sys.exit("Use: python train_tgcn.py <dataset>")
cfg.dataset = dataset    
    


# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus(cfg.dataset)

features = sp.identity(features.shape[0])  # featureless
features = preprocess_features(features)


support = [preprocess_adj(adj)]


t_features = tf.SparseTensor(*features)
t_y_train = tf.convert_to_tensor(y_train)
t_y_val = tf.convert_to_tensor(y_val)
t_y_test = tf.convert_to_tensor(y_test)
tm_train_mask = tf.convert_to_tensor(train_mask)

tm_val_mask = tf.convert_to_tensor(val_mask)
tm_test_mask = tf.convert_to_tensor(test_mask)

t_support = []
for i in range(len(support)):
    t_support.append(tf.cast(tf.SparseTensor(*support[i]), dtype=tf.float64))


# Create model
model = GCN(input_dim=features[2][1], output_dim=y_train.shape[1], num_features_nonzero=features[1].shape)



# Loss and optimizer
optimizer = optimizers.Adam(lr=cfg.learning_rate)

cost_val = []

for epoch in range(cfg.epochs):
    
    t = time.time()
    with tf.GradientTape() as tape:
        _, loss, acc = model((t_features, t_y_train, tm_train_mask, t_support))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    _, val_loss, val_acc = model((t_features, t_y_val, tm_val_mask, t_support), training=False)
    cost_val.append(val_loss)
    
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss),
          "train_acc=", "{:.5f}".format(acc), "val_loss=", "{:.5f}".format(val_loss),
          "val_acc=", "{:.5f}".format(val_acc), "time=", "{:.5f}".format(time.time() - t))
    
    if epoch > cfg.early_stopping and cost_val[-1] > np.mean(cost_val[-(cfg.early_stopping+1):-1]):
        print("Early stopping...")
        break

def evaluate(features, y, mask, support):
    t = time.time()
    
    pred, test_loss, test_acc = model((features, y, mask, support), training=False)
    
    
    return test_loss, test_acc, pred, np.argmax(y, axis=1), time.time() - t


test_cost, test_acc, pred, labels, test_duration = evaluate(t_features, t_y_test, tm_test_mask, t_support)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))


test_pred = []
test_labels = []

for i in range(len(test_mask)):
    if test_mask[i]:
        test_pred.append(pred[i])
        test_labels.append(labels[i])

print("Average Test Precision, Recall and F1-Score...")
print(metrics.precision_recall_fscore_support(test_labels, test_pred, average='micro'))


