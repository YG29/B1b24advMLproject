from keras.datasets import mnist
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np
import ssl


try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

# mnist.load_data(path="mnist.npz")
# (X_train, y_train), (X_test, y_test) = mnist.load_data()

# X_train, y_train = torch.tensor(X_train), torch.tensor(y_train)
# X_test, y_test = torch.tensor(X_test), torch.tensor(y_test)

# print(X_train.data.shape)


# train_data = (X_train, y_train)
# test_data = (X_test, y_test)

train_data = datasets.MNIST(
    root = 'data',
    train = True, 
    transform = ToTensor(),
    download = True
)

test_data = datasets.MNIST(
    root = 'data',
    train = False, 
    transform = ToTensor(),
    download = True
)

# print(train_data)
#print(train_data.shape())
# print(test_data)
#print(test_data.shape())

loaders = {

    'train': DataLoader(train_data,
                        batch_size = 100, 
                        shuffle = True),

    'test': DataLoader(test_data,
                        batch_size = 100, 
                        shuffle = True)

}

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size = 3) # takes in 1 channel, outputs 10 channels
        self.conv2 = nn.Conv2d(10, 20, kernel_size = 3) # takes in 10 channels, outputs 20 channels
        self.fc1 = nn.Linear(500, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))   
        x = F.relu(F.max_pool2d(self.conv2(x), 2))  
        x = x.view(-1, 500) # Flattens data for the linear layers
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.softmax(self.fc2(x))

        return x
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
lossFunc = nn.CrossEntropyLoss()

def train(epoch):
    model.train()
    
    for batch_idx, (data, target) in enumerate(loaders['train']):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data) # predicts the class of the given data according to the current state of the model
        loss = lossFunc(output, target)
        loss.backward() # back propogates according to the loss
        optimizer.step()
        if batch_idx % 20 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)} / {len(loaders["train"].dataset)} ({100. * batch_idx / len(loaders["train"]):.0f}%)]\t Loss: {loss.item():.6f}')

def test():
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in loaders['test']:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += lossFunc(output, target).item()
            prediction = output.argmax(dim=1, keepdim = True)
            correct += prediction.eq(target.view_as(prediction)).sum().item()

    test_loss /= len(loaders['test'].dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(loaders["test"].dataset)} ({100. * correct / len(loaders["test"].dataset):.0f}%\n)')

for epoch in range(1, 11):
    train(epoch)
    test()