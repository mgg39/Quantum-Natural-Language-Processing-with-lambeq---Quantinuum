#numpy model training
import os
import warnings

warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

import numpy as np

BATCH_SIZE = 30
EPOCHS = 1000
SEED = 2

#input data
def read_data(filename):
    labels, sentences = [], []
    with open(filename) as f:
        for line in f:
            t = int(line[0])
            labels.append([t, 1-t])
            sentences.append(line[1:].strip())
    return labels, sentences


train_labels, train_data = read_data('./MC1.TXT')
val_labels, val_data = read_data('./MC1.TXT')