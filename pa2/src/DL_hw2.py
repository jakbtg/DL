import time
import torch
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


#--- hyperparameters ---
N_EPOCHS = 30
BATCH_SIZE_TRAIN = 100
BATCH_SIZE_TEST = 100
LR = 0.2
OPTIMIZER = 'SGD' # SGD, Adam, Adagrad, SGD_momentum


#--- fixed constants ---
NUM_CLASSES = 24
DATA_DIR = '../data/sign_mnist_%s'



# --- Dataset initialization ---

# We transform image files' contents to tensors
# Plus, we can add random transformations to the training data if we like
# Think on what kind of transformations may be meaningful for this data.
# Eg., horizontal-flip is definitely a bad idea for sign language data.
# You can use another transformation here if you find a better one.
train_transform = transforms.Compose([
                                        #transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor()])
test_transform = transforms.Compose([transforms.ToTensor()])

train_set = datasets.ImageFolder(DATA_DIR % 'train', transform=train_transform)
dev_set   = datasets.ImageFolder(DATA_DIR % 'dev',   transform=test_transform)
test_set  = datasets.ImageFolder(DATA_DIR % 'test',  transform=test_transform)


# Create Pytorch data loaders
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=BATCH_SIZE_TEST, shuffle=True)
dev_loader = torch.utils.data.DataLoader(dataset=dev_set, batch_size=BATCH_SIZE_TEST, shuffle=True)



#--- model ---
class CNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(CNN, self).__init__()
        
        # WRITE CODE HERE
        # CNN layers: 2 conv layers, 2 pooling layers    
        # FFNN layers: 3 linear layers that learn the classification
        # Using ReLU as the activation function
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=12, kernel_size=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=12, out_channels=32, kernel_size=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(32*4*4, 120),
            nn.ReLU(),
            nn.Linear(120, 96),
            nn.ReLU(),
            nn.Linear(96, num_classes)
        )


    def forward(self, x):
        # WRITE CODE HERE
        x = self.cnn_layers(x)
        x = self.flatten(x)
        x = self.linear_relu_stack(x)
        return x




#--- set up ---
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model = CNN().to(device)
print(model)

# WRITE CODE HERE
if OPTIMIZER == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=LR)
elif OPTIMIZER == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
elif OPTIMIZER == 'Adagrad':
    optimizer = optim.Adagrad(model.parameters(), lr=LR)
elif OPTIMIZER == 'SGD_momentum':
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.2)
loss_function = nn.CrossEntropyLoss()


#--- training ---
starting_time = time.time()
actual_epochs = 0
prev_avg_dev_loss = float('inf')
count = 0

for epoch in range(N_EPOCHS):
    train_loss = 0
    train_correct = 0
    total = 0
    
    training_accuracies = []
    avg_training_accuracy = 0
    
    for batch_num, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # TRAINING STEPS
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predictions = torch.max(output.data, 1)
        train_correct += (predictions == target).sum().item()
        total += target.size(0)

        print('Training: Epoch %d - Batch %d/%d: Loss: %.4f | Train Acc: %.3f%% (%d/%d)' % 
              (epoch, batch_num+1, len(train_loader), train_loss / (batch_num + 1), 
               100. * train_correct / total, train_correct, total))
        
        training_accuracies.append(100. * train_correct / total)

    avg_training_accuracy = np.mean(training_accuracies)
    
    # WRITE CODE HERE
    # Please implement early stopping here.
    # You can try different versions, simplest way is to calculate the dev error and
    # compare this with the previous dev error, stopping if the error has grown.
    model.eval()
    dev_loss = 0
    dev_correct = 0
    total = 0

    dev_accuracies = []
    avg_dev_accuracy = 0
    
    dev_losses = []
    avg_dev_loss = 0
    
    for batch_num, (data, target) in enumerate(dev_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = loss_function(output, target)
        dev_loss += loss.item()
        # dev_losses.append(dev_loss / (batch_num + 1))
        dev_losses.append(loss.item())
        _, predictions = torch.max(output.data, 1)
        dev_correct += (predictions == target).sum().item()
        total += target.size(0)

        print('Validation: Epoch %d - Batch %d/%d: Loss: %.4f | Dev Acc: %.3f%% (%d/%d)' % 
              (epoch, batch_num+1, len(dev_loader), dev_loss / (batch_num + 1), 
               100. * dev_correct / total, dev_correct, total))
        
        dev_accuracies.append(100. * dev_correct / total)

    avg_dev_accuracy = np.mean(dev_accuracies)
    avg_dev_loss = np.mean(dev_losses)


    # if epoch > 10 and avg_dev_loss > prev_avg_dev_loss:
    if avg_dev_loss > prev_avg_dev_loss:
        count += 1
        print(f'Count: {count}')
        if count == 2:
            print('Early stopping')
            print(f'Previous average dev loss: {prev_avg_dev_loss:.2f} - Current average dev loss: {avg_dev_loss:.2f}')
            break
    
    print(f'Previous average dev loss: {prev_avg_dev_loss:.2f} - Current average dev loss: {avg_dev_loss:.2f}')
    prev_avg_dev_loss = avg_dev_loss

    actual_epochs += 1




#--- test ---
test_loss = 0
test_correct = 0
total = 0

test_accuracies = []
avg_test_accuracy = 0

model.eval()
with torch.no_grad():
    for batch_num, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        
        # WRITE CODE HERE
        # Compute predictions
        output = model(data)
        loss = loss_function(output, target)
        test_loss += loss.item()
        _, predictions = torch.max(output.data, 1)
        test_correct += (predictions == target).sum().item()
        total += target.size(0)

        print('Evaluating: Batch %d/%d: Loss: %.4f | Test Acc: %.3f%% (%d/%d)' % 
              (batch_num+1, len(test_loader), test_loss / (batch_num + 1), 
               100. * test_correct / total, test_correct, total))
        
        test_accuracies.append(100. * test_correct / total)

    avg_test_accuracy = np.mean(test_accuracies)


text = 'CNN results with the following parameters:\n'

text += f'Max Epochs: {N_EPOCHS}\n'
text += f'Actual Epochs: {actual_epochs}\n'
text += f'Batch size: {BATCH_SIZE_TRAIN}\n'
text += f'Learning rate: {LR}\n'
text += f'Optimizer: {OPTIMIZER}\n'

text += f'Training accuracy: {avg_training_accuracy:.2f}\n'
text += f'Dev accuracy: {avg_dev_accuracy:.2f}\n'
text += f'Test accuracy: {avg_test_accuracy:.2f}\n'

elapsed_time = time.time() - starting_time
text += f'Total training time: {time.strftime("%Hh:%Mm:%Ss", time.gmtime(elapsed_time))}\n'
text += '-' * 50 + '\n'
print(text)

# save results to file if some of the hyperparameters were changed
with open('results.txt', 'a') as f:
    f.write(text)

