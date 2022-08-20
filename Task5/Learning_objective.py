#numpy model training
import os
import warnings

warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

import numpy as np
from lambeq import BobcatParser
from discopy import grammar
from discopy import Dim
from lambeq import AtomicType, SpiderAnsatz
#from pytket.extensions.qiskit import AerBackend
from lambeq import TketModel

BATCH_SIZE =42
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


train_labels, train_data = read_data('Task5/train_data.txt')
#print(train_data[:8])
#print(train_labels[:8])

val_labels, val_data = read_data('Task5/val_data.txt')
#print(val_data[:8])
#print(val_labels[:8])


#parser
parser = BobcatParser(root_cats=('NP', 'N'), verbose='text')
train_diagrams = parser.sentences2diagrams(train_data, suppress_exceptions=True)

train_diagrams = [
    diagram.normal_form()
    for diagram in train_diagrams if diagram is not None
]

train_labels = [
    label for (diagram, label)
    in zip(train_diagrams, train_labels)
    if diagram is not None]

#train_diagrams[0].draw(figsize=(9, 5), fontsize=12)
