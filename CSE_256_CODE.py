#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import torch
import torch.nn as nn

import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from torchtext.data.utils import get_tokenizer

from gensim.models import KeyedVectors
import torch.nn.functional as F
import gzip
import shutil

# Set device to GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# install the libraries if not installed.
'''
!pip install torch torchvision torchtext
!pip install torch torchtext spacy
!python -m spacy download en_core_web_sm
!pip install portalocker
!pip install gensim requests

'''

# Download the dataset if required

get_ipython().system('wget http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz')
get_ipython().system('tar -xvzf rt-polaritydata.tar.gz')

def load_data_and_labels(positive_file, negative_file):
    """
    Load the data and labels from the given files.

    Args:
        positive_file (str): Path to the file containing positive examples.
        negative_file (str): Path to the file containing negative examples.

    Returns:
        tuple: A tuple containing a list of texts and a list of corresponding labels.
    """
    positive_examples = open(positive_file, 'r', encoding='latin-1').readlines()
    negative_examples = open(negative_file, 'r', encoding='latin-1').readlines()
    
    # Clean the data by removing whitespace
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = [s.strip() for s in negative_examples]
    
    # Create labels (1 for positive, 0 for negative)
    positive_labels = [1 for _ in positive_examples]
    negative_labels = [0 for _ in negative_examples]
    
    return positive_examples + negative_examples, positive_labels + negative_labels

# Load data
positive_file = 'rt-polaritydata/rt-polarity.pos'
negative_file = 'rt-polaritydata/rt-polarity.neg'
texts, labels = load_data_and_labels(positive_file, negative_file)

# Tokenize the text
from collections import Counter
from itertools import chain

# Use basic_english tokenizer from torchtext
tokenizer = get_tokenizer("basic_english")

# Create a vocabulary based on word frequency
counter = Counter(chain(*[tokenizer(text) for text in texts]))
vocab = {word: idx + 1 for idx, (word, _) in enumerate(counter.most_common())}
vocab["<PAD>"] = 0  # Add padding token

def text_to_sequence(text, vocab, max_length):
    """
    Convert a text to a sequence of token IDs.

    Args:
        text (str): The input text.
        vocab (dict): A dictionary mapping words to their corresponding IDs.
        max_length (int): The maximum length of the sequence.

    Returns:
        list: A list of token IDs.
    """
    tokens = tokenizer(text)
    sequence = [vocab[token] if token in vocab else 0 for token in tokens]
    if len(sequence) < max_length:
        sequence += [0] * (max_length - len(sequence))
    return sequence[:max_length]

# Determine the maximum sequence length
max_sequence_length = max(len(tokenizer(text)) for text in texts)

# Convert all texts to sequences of token IDs
sequences = [text_to_sequence(text, vocab, max_sequence_length) for text in texts]

# Convert to PyTorch tensors
data = torch.tensor(sequences, dtype=torch.long)
labels = torch.tensor(labels, dtype=torch.long)

# Split data into training and testing sets
dataset = torch.utils.data.TensorDataset(data, labels)
train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=50, shuffle=False)

# Path to the local file for the Word2Vec model file, which contains word embeddings trained on the Google News dataset.

# Download the Google News pre-trained Word2Vec model as required

get_ipython().system('wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"')


local_file_path = 'GoogleNews-vectors-negative300.bin.gz' # Modify the path accordingly to locate the file 

# Extract the .gz file
with gzip.open(local_file_path, 'rb') as f_in:
    with open('GoogleNews-vectors-negative300.bin', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

# Load the Word2Vec model
word2vec_path = 'GoogleNews-vectors-negative300.bin' # Modify the path accordingly
word2vec = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

# Create the embedding matrix
embedding_dim = 300
embedding_matrix = np.zeros((len(vocab), embedding_dim))
for word, idx in vocab.items():
    if word in word2vec:
        embedding_matrix[idx] = word2vec[word]
    else:
        embedding_matrix[idx] = np.random.normal(size=(embedding_dim,))

embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float)

''' Here, different architectures of convolution neural network were tried by varying the number of conv layers, hidden layers, 
activation functions like Relu, Sigmoid, Softmax etc, ading batch normalization, dropout and pooling layers like average
pooling and max pooling. Some of the models are given below.'''


# This architecture consists of five convolution layers with adagrad optimizer, learning rate=0.005, weight_decay=0.001

class CNN_1(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes=2):
        super(CNN_1, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)  # Random initialization

        # Five convolutional layers
        self.conv1 = nn.Conv2d(1, 128, (3, embed_dim))
        self.conv2 = nn.Conv2d(1, 128, (3, embed_dim))
        self.conv3 = nn.Conv2d(1, 128, (3, embed_dim))
        self.conv4 = nn.Conv2d(1, 128, (3, embed_dim))
        self.conv5 = nn.Conv2d(1, 128, (3, embed_dim))

        # Fully connected layer
        self.fc = nn.Linear(128 * 5, num_classes)  # Concatenated output size of five sets

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)  # Add channel dimension

        # Apply five convolutional layers with ReLU activation and max pooling
        x1 = torch.relu(self.conv1(x)).squeeze(3)
        x1 = torch.max_pool1d(x1, x1.size(2)).squeeze(2)

        x2 = torch.relu(self.conv2(x)).squeeze(3)
        x2 = torch.max_pool1d(x2, x2.size(2)).squeeze(2)

        x3 = torch.relu(self.conv3(x)).squeeze(3)
        x3 = torch.max_pool1d(x3, x3.size(2)).squeeze(2)

        x4 = torch.relu(self.conv4(x)).squeeze(3)
        x4 = torch.max_pool1d(x4, x4.size(2)).squeeze(2)

        x5 = torch.relu(self.conv5(x)).squeeze(3)
        x5 = torch.max_pool1d(x5, x5.size(2)).squeeze(2)

        # Concatenate all features from the convolutional layers
        x = torch.cat([x1, x2, x3, x4, x5], 1)

        logits = self.fc(x)
        return logits 
        
        
model1 = CNN_1(len(vocab), embedding_dim, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer1 = optim.Adagrad(model1.parameters(), lr=0.005, weight_decay=0.001)



# This architecture consists of three conv layers with Adadelta optimizer, learning rate 0.001, weight decay = 0.0001


class CNN_2(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes=2):
        super(CNN_2, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)  # Random initialization

        # Three convolutional layers
        self.conv1 = nn.Conv2d(1, 128, (3, embed_dim))
        self.conv2 = nn.Conv2d(1, 128, (4, embed_dim))
        self.conv3 = nn.Conv2d(1, 128, (5, embed_dim))

        # Fully connected layer
        self.fc = nn.Linear(128 * 3, num_classes)  # Concatenated output size of three sets

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)  # Add channel dimension

        # Apply three convolutional layers with ReLU activation and max pooling
        x1 = torch.relu(self.conv1(x)).squeeze(3)
        x1 = torch.max_pool1d(x1, x1.size(2)).squeeze(2)

        x2 = torch.relu(self.conv2(x)).squeeze(3)
        x2 = torch.max_pool1d(x2, x2.size(2)).squeeze(2)

        x3 = torch.relu(self.conv3(x)).squeeze(3)
        x3 = torch.max_pool1d(x3, x3.size(2)).squeeze(2)

        # Concatenate all features from the convolutional layers
        x = torch.cat([x1, x2, x3], 1)

        logits = self.fc(x)
        return logits
    
    
model2 = CNN_2(len(vocab), embedding_dim, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer2 = optim.Adadelta(model2.parameters(), lr=0.001, weight_decay=0.0001)
       

    
# This architecture consists of ten convolution layers, three fully connnected layers, Adam optimizer
# learning rate=0.003, weight_decay=0.00001

class CNN_3(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes=2):
        super(CNN_3, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)  # Random initialization

        # Ten convolutional layers with different filter sizes
        self.conv1 = nn.Conv2d(1, 128, (2, embed_dim))
        self.conv2 = nn.Conv2d(1, 128, (3, embed_dim))
        
        self.conv3 = nn.Conv2d(1, 128, (4, embed_dim))
        self.conv4 = nn.Conv2d(1, 128, (5, embed_dim))
        
        self.conv5 = nn.Conv2d(1, 128, (6, embed_dim))
        self.conv6 = nn.Conv2d(1, 128, (7, embed_dim))
        
        self.conv7 = nn.Conv2d(1, 128, (8, embed_dim))
        self.conv8 = nn.Conv2d(1, 128, (9, embed_dim))
        
        self.conv9 = nn.Conv2d(1, 128, (10, embed_dim))
        self.conv10 = nn.Conv2d(1, 128, (11, embed_dim))

        # Three fully connected layers
        self.fc1 = nn.Linear(128 * 10, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)  # Add channel dimension

        # Apply ten convolutional layers with ReLU activation and max pooling
        conv_layers = [
            self.conv1, self.conv2, self.conv3, self.conv4, self.conv5,
            self.conv6, self.conv7, self.conv8, self.conv9, self.conv10
        ]

        conv_outputs = []
        for conv in conv_layers:
            conv_out = torch.relu(conv(x)).squeeze(3)
            conv_out = torch.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            conv_outputs.append(conv_out)

        # Concatenate all features from the convolutional layers
        x = torch.cat(conv_outputs, 1)

        # Apply fully connected layers with ReLU activation
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits
    
model3 = CNN_3len(vocab), embedding_dim, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer3 = optim.Adam(model3.parameters(), lr=0.003, weight_decay=0.00001)



# This architecture consists of eight convolution layers with AdamW optimizer, relu activation, max pooling,
# learning rate = 0.002, weight_decay=0.00001

class CNN_R(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes=2):
        """
        Initialize the CNN model with random initialization.

        Args:
            vocab_size (int): The size of the vocabulary.
            embed_dim (int): The dimension of the embeddings.
            num_classes (int): The number of output classes.
        """
        super(CNN_R, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)  # Random initialization model
        
        # self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False) # For Non-Static model
        # self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True) # For Static model

        # Adding convolutional layers
        
        self.conv1 = nn.Conv2d(1, 256, (3, embed_dim))
        self.conv2 = nn.Conv2d(1, 256, (3, embed_dim))
        
        self.conv3 = nn.Conv2d(1, 256, (3, embed_dim))
        self.conv4 = nn.Conv2d(1, 256, (3, embed_dim))
        
        self.conv5 = nn.Conv2d(1, 256, (3, embed_dim))
        self.conv6 = nn.Conv2d(1, 256, (3, embed_dim))
        
        self.conv7 = nn.Conv2d(1, 256, (3, embed_dim))
        self.conv8 = nn.Conv2d(1, 256, (3, embed_dim))

        # Fully connected layer
        self.fc = nn.Linear(256 * 8, num_classes)  # Concatenated output size of eight sets

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, sequence_length).

        Returns:
            Tensor: Output logits of the model.
        """
        x = self.embedding(x).unsqueeze(1)  # Add channel dimension

        # Apply convolutional layers with ReLU activation and max pooling
        x1 = torch.relu(self.conv1(x)).squeeze(3)
        x1 = torch.max_pool1d(x1, x1.size(2)).squeeze(2)

        x2 = torch.relu(self.conv2(x)).squeeze(3)
        x2 = torch.max_pool1d(x2, x2.size(2)).squeeze(2)

        x3 = torch.relu(self.conv3(x)).squeeze(3)
        x3 = torch.max_pool1d(x3, x3.size(2)).squeeze(2)

        x4 = torch.relu(self.conv4(x)).squeeze(3)
        x4 = torch.max_pool1d(x4, x4.size(2)).squeeze(2)

        x5 = torch.relu(self.conv5(x)).squeeze(3)
        x5 = torch.max_pool1d(x5, x5.size(2)).squeeze(2)

        x6 = torch.relu(self.conv6(x)).squeeze(3)
        x6 = torch.max_pool1d(x6, x6.size(2)).squeeze(2)

        x7 = torch.relu(self.conv7(x)).squeeze(3)
        x7 = torch.max_pool1d(x7, x7.size(2)).squeeze(2)

        x8 = torch.relu(self.conv8(x)).squeeze(3)
        x8 = torch.max_pool1d(x8, x8.size(2)).squeeze(2)

        x = torch.cat([x1, x2, x3, x4, x5, x6, x7, x8], 1)

        # Fully connected layer to output logits
        logits = self.fc(x)
        
        return logits

num_classes = 2
model2 = CNN_R(len(vocab), embedding_dim, num_classes).to(device)

# Define loss function and optimizer

''' Here, the learning rate was varied as 0.1,0.001 0.0001,0.01 etc and the weight decay is varied as 0.01,0.001,0.0001,0.00001,
0.000001 etc, number of epochs as 30,50,70,100 etc. Also, different optimizers were tried like Stochastic gradient descent ,
Adam, AdamW, Root Mean Squared Propagation, Adadelta, Adagrad etc. 
'''

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model2.parameters(), lr=0.002, weight_decay=0.00001)

# Training loop, modify number of epoch as required
num_epochs = 50
for epoch in range(num_epochs):
    model2.train()
    total_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model2(inputs)
        
        # Compute loss
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Accumulate loss
        total_loss += loss.item()
    
    # Print average loss for the epoch
    print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}')

# Evaluate the model

model2.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model2(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(targets.cpu().numpy())

# Compute and print accuracy

accuracy = accuracy_score(all_labels, all_preds)
print(f'Test Accuracy: {accuracy}')

