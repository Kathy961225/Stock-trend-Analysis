class CONFIG_TGCN(object):
    """docstring for CONFIG"""
    def __init__(self):
        super(CONFIG_TGCN, self).__init__()
        
        self.dataset = 'LC'
        self.learning_rate = 0.02  
        self.epochs  = 100  # Number of epochs to train.
        self.hidden1 = 200  # Number of units in hidden layer 1.
        self.dropout = 0.5  # Dropout rate (1 - keep probability).
        self.weight_decay = 0.   # Weight for L2 loss on embedding matrix.
        self.early_stopping = 10 # Tolerance for early stopping (# of epochs).
        
        
        
class CONFIG_BERT(object):
    """docstring for CONFIG"""
    def __init__(self):
        super(CONFIG_BERT, self).__init__()
        
        self.dataset = 'LC'
        self.learning_rate = 0.02  
        self.epochs  = 20  # Number of epochs to train.
        self.max_length = 20 # Max length of text.
        self.num_labels = 3
        self.batch_size = 128
        self.learning_rate = 1e-5