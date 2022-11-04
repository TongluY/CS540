# Tonglu Yang
# cs540 hw6

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.


def get_data_loader(training = True):
    """
    TODO: implement this function.

    INPUT: 
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """
    custom_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    train_set=datasets.FashionMNIST('./data',train=True, download=True,transform=custom_transform)
    test_set=datasets.FashionMNIST('./data', train=False, transform=custom_transform)
    if training: return torch.utils.data.DataLoader(train_set, batch_size = 64)
    else: return torch.utils.data.DataLoader(test_set, batch_size = 64, shuffle = False)


def build_model():
    """
    TODO: implement this function.

    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    return model

def train_model(model, train_loader, criterion, T):
    """
    TODO: implement this function.

    INPUT: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy 
        T - number of epochs for training

    RETURNS:
        None
    """
    
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()
    for epoch in range(T):
        loss = 0.0
        correct = 0.0
        total = 0
        for i, data in enumerate(train_loader, 0):
            images, labels = data
            opt.zero_grad()
            outputs = model(images)
            loss_ = criterion(outputs, labels)
            loss_.backward()
            opt.step()
            loss += loss_.item() 
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Train Epoch: ', epoch, ' Accuracy: ', int(correct), '/', int(total),'(', "%.2f" % round(100 * correct / 60000,2), '%) Loss: ', "%.3f" % round(loss/total*64,3))
          #  loss = 0.0

def evaluate_model(model, test_loader, criterion, show_loss = True):
    """
    TODO: implement this function.

    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 

    RETURNS:
        None
    """
    model.eval()
    loss = 0.0
    correct = 0.0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            loss_ = criterion(outputs, labels)
            loss += loss_.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    if show_loss: print("Average loss: ", str("%.4f" % (loss / len(test_loader))))
    print("Accuracy: ", str("%.2f" % (100 * correct / total)), "%")

def predict_label(model, test_images, index):
    """
    TODO: implement this function.

    INPUT: 
        model - the trained model
        test_images   -  test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1


    RETURNS:
        None
    """
    class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot']
    prob = F.softmax(model(test_images), dim = 1)
    rl = sorted(prob.tolist()[index],reverse = True)
    i0 = rl.index(rl[0])
    i1 = rl.index(rl[1])
    i2 = rl.index(rl[2])
    print(class_names[i0], ": ", "%.2f" % round(rl[i0]*100, 2), "%")
    print(class_names[i1], ": ", "%.2f" % round(rl[i1]*100, 2), "%")
    print(class_names[i2], ": ", "%.2f" % round(rl[i2]*100, 2), "%")


if __name__ == '__main__':
    '''
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    '''
    criterion = nn.CrossEntropyLoss()
    train_loader = get_data_loader()
    test_loader = get_data_loader(False)
    model = build_model()
    train_model(model,train_loader, criterion, 5)
    # evaluate_model(model, test_loader, criterion, show_loss = False)
    evaluate_model(model, test_loader, criterion, show_loss = True)
    pred_set, _ = next(iter(test_loader))
    predict_label(model, pred_set, 1)