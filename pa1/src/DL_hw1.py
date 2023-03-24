"""
   Introduction to Deep Learning (LDA-T3114)
   Skeleton Code for Assignment 1: Sentiment Classification on a Feed-Forward Neural Network

   Hande Celikkanat & Miikka Silfverberg
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import time

#ATTENTION: If necessary, add the paths to your data_semeval.py and paths.py here:
#import sys
#sys.path.append('</Users/jak/Documents/Uni/DL - Helsinki/Esercizi/intro_to_dl_assignment_1/src/data_semeval.py>')
from data_semeval import *
from paths import data_dir


#--- hyperparameters ---
N_CLASSES = len(LABEL_INDICES)
N_EPOCHS = 20
LEARNING_RATE = 0.1
BATCH_SIZE = 15
REPORT_EVERY = 1
IS_VERBOSE = False

#--- hidden layers hyperparameters ---
HIDDEN_SIZE_1 = 64
HIDDEN_SIZE_2 = 64


def make_bow(tweet, indices):
    feature_ids = list(indices[tok] for tok in tweet['BODY'] if tok in indices)
    bow_vec = torch.zeros(len(indices))
    bow_vec[feature_ids] = 1
    return bow_vec.view(1, -1)

def generate_bow_representations(data):
    vocab = set(token for tweet in data['training'] for token in tweet['BODY'])
    vocab_size = len(vocab) 
    indices = {w:i for i, w in enumerate(vocab)}
  
    for split in ["training","development.input","development.gold",
                  "test.input","test.gold"]:
        for tweet in data[split]:
            tweet['BOW'] = make_bow(tweet,indices)

    return indices, vocab_size

# Convert string label to pytorch format.
def label_to_idx(label):
    return torch.LongTensor([LABEL_INDICES[label]])



#--- model ---
class FFNN(nn.Module):
    # Feel free to add whichever arguments you like here.
    def __init__(self, vocab_size, n_classes, extra_arg_1=None, extra_arg_2=None):
        super(FFNN, self).__init__()

        # --------------
        # CODE HERE
        # --------------

        # Define the sizes of the layers of the network
        self.input_size = vocab_size
        self.hidden_size_1 = extra_arg_1
        self.hidden_size_2 = extra_arg_2
        self.output_size = n_classes

        # Define the layers of the network
        # - fc1 = fully connected first layer: input_size -> hidden_size_1
        # - relu1 = non-linearity (or tanh)
        # - fc2 = fully connected second layer: hidden_size_1 -> hidden_size_2 (only if extra_arg_2 is not None)
        # - relu2 = non-linearity (or tanh)
        # - out = output layer: hidden_size_2 -> output_size

        self.fc1 = nn.Linear(self.input_size, self.hidden_size_1)
        self.relu1 = nn.ReLU()
        # self.tanh1 = nn.Tanh()
        if self.hidden_size_2:
            self.fc2 = nn.Linear(self.hidden_size_1, self.hidden_size_2)
            self.relu2 = nn.ReLU()
            # self.tanh2 = nn.Tanh()
            self.out = nn.Linear(self.hidden_size_2, self.output_size)
        else:
            self.out = nn.Linear(self.hidden_size_1, self.output_size)


        

    def forward(self, x):
        
        # --------------
        # CODE HERE
        # --------------

        # Define the forward pass of the network as a pipeline
        output = self.fc1(x)
        output = self.relu1(output)
        # output = self.tanh1(output)
        
        # If there is a second hidden layer, add it to the pipeline
        if self.hidden_size_2:
            output = self.fc2(output)
            output = self.relu2(output)
            # output = self.tanh2(output)
        
        output = self.out(output)

        # Need to apply log_softmax to the output because we are using NLLLoss
        output = F.log_softmax(output, dim=1)
        # If you are using CrossEntropyLoss, you don't need to apply log_softmax
        return output




#--- data loading ---
data = read_semeval_datasets(data_dir)
indices, vocab_size = generate_bow_representations(data)



#--- set up ---
model = FFNN(vocab_size, N_CLASSES, HIDDEN_SIZE_1, HIDDEN_SIZE_2)
# Negative Log Likelihood Loss
loss_function = nn.NLLLoss()
# Cross Entropy Loss
# loss_function = nn.CrossEntropyLoss()
# SGD optimizer
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE) 



#--- training ---
total_elapsed_time = time.time()
for epoch in range(N_EPOCHS):
    total_loss = 0
    # Generally speaking, it's a good idea to shuffle your
    # datasets once every epoch.
    random.shuffle(data['training'])

    starting_time_epoch = time.time()

    for i in range(int(len(data['training'])/BATCH_SIZE)):
        minibatch = data['training'][i*BATCH_SIZE:(i+1)*BATCH_SIZE]

        # --------------
        # CODE HERE
        # --------------

        # Prepare input matrix and target vector
        # - input_matrix = matrix of size BATCH_SIZE x vocab_size
        # - target_vector = vector of size BATCH_SIZE
        # Use torch.cat() to concatenate the tensors in the minibatch 
        input_matrix = torch.cat([tweet['BOW'] for tweet in minibatch])
        target_vector = torch.cat([label_to_idx(tweet['SENTIMENT']) for tweet in minibatch])

        # input_matrix = torch.zeros(BATCH_SIZE, vocab_size)
        # target_vector = torch.zeros(BATCH_SIZE, dtype=torch.long)
        # for j in range(BATCH_SIZE):
        #     input_matrix[j] = minibatch[j]['BOW']
        #     target_vector[j] = label_to_idx(minibatch[j]['SENTIMENT'])

        # Compute predictions and loss
        predictions = model(input_matrix)
        loss = loss_function(predictions, target_vector)

        # Update total loss
        total_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad() # in order to avoid accumulating gradients
        loss.backward()
        optimizer.step() # update weights of the model
                      
    if ((epoch+1) % REPORT_EVERY) == 0:
        print('epoch: %d, loss: %.4f' % (epoch+1, total_loss*BATCH_SIZE/len(data['training'])))
        # print(f'epoch {epoch + 1}: {time.time() - starting_time_epoch:.2f} seconds, total time: {time.time() - total_elapsed_time:.2f} seconds')




#--- test set ---
correct = 0
with torch.no_grad():
    for tweet in data['test.gold']:
        gold_class = label_to_idx(tweet['SENTIMENT'])

        # --------------
        # CODE HERE
        # --------------

        # Compute predictions
        predictions = model(tweet['BOW'])

        # Compute accuracy
        predicted = torch.argmax(predictions)
        if predicted == gold_class:
            correct += 1

        if IS_VERBOSE:
            print('TEST DATA: %s, GOLD LABEL: %s, GOLD CLASS %d, OUTPUT: %d' % 
                 (' '.join(tweet['BODY'][:-1]), tweet['SENTIMENT'], gold_class, predicted))

    print('test accuracy: %.2f' % (100.0 * correct / len(data['test.gold'])))

#--- development set ---
correct = 0
with torch.no_grad():
    for tweet in data['development.gold']:
        gold_class = label_to_idx(tweet['SENTIMENT'])

        # --------------
        # CODE HERE
        # --------------

        # Compute predictions
        predictions = model(tweet['BOW'])

        # Compute accuracy
        predicted = torch.argmax(predictions)
        if predicted == gold_class:
            correct += 1

        if IS_VERBOSE:
            print('DEV DATA: %s, GOLD LABEL: %s, GOLD CLASS %d, OUTPUT: %d' % 
                 (' '.join(tweet['BODY'][:-1]), tweet['SENTIMENT'], gold_class, predicted))

    print('dev accuracy: %.2f' % (100.0 * correct / len(data['development.gold'])))