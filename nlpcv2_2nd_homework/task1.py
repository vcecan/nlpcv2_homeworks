import torch
from torch import nn as nn
from torch.utils.data import Dataset
from torch import optim
from torch.nn import functional as F
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

DEVICE = torch.device("cpu" )
x=pd.read_csv("diabetes.csv")
#print (x)

class Dataset(Dataset): # task 1.1 implementam DataSet class

    def __init__(self,path):
        self.dataset = pd.read_csv(path).values

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return self.dataset[i,:-1], self.dataset[i,-1]

class Net(nn.Module): ## 1.2 implementarea clasei retelei neuronale

    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features
        #fc -fully conncted
        self.fc1 =nn.Linear(self.n_features,16)
        self.fc2 =nn.Linear(16,32)
        self.fc3 =nn.Linear(32,1)

    def forward (self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = torch.sigmoid(self.fc3(out))

        return out

corect=0
def training_loop(n_epochs, optimizer, model, loss_fn, train_loader,
                  test_loader,print_plot=True):
    global corect
    train_accuracy = np.zeros(n_epochs)
    test_accuracy = np.zeros(n_epochs)

    train_loss = np.zeros(n_epochs)
    test_loss = np.zeros(n_epochs)

    for epoch in range(n_epochs):
        model.train()
        total = 0
        correct = 0  #pentru acuratete
        current_train_loss = 0.0

    #train_loader - date de antrenare
        for examples,labels in train_loader:
            examples = examples.to(DEVICE).float()
            labels = labels.to(DEVICE).float()

            labels = labels.unsqueeze(1)#adaugam 1 dimensiune

            predicted  = model(examples)

            loss = loss_fn(predicted, labels)
            current_train_loss += loss

            predicted = torch.round(predicted)
            total+=labels.shape[0]
            correct +=(predicted==labels).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_accuracy[epoch] = correct/total
        train_loss[epoch] = current_train_loss/total

        model.eval()
        total = 0
        correct = 0  # pentru acuratete
        current_test_loss = 0.0

        for examples,labels in test_loader:

            examples = examples.to(DEVICE).float()
            labels = labels.to(DEVICE).float()

            labels = labels.unsqueeze(1)  # adaugam 1 dimensiun

            predicted = model(examples)
            loss = loss_fn(predicted, labels)
            current_test_loss += loss

            predicted = torch.round(predicted)
            total += labels.shape[0]
            correct += (predicted == labels).sum()

        test_accuracy[epoch] = correct / total
        test_loss[epoch] = current_test_loss / total

        if(epoch+1)%10==0:
            print(f'Epoch {epoch+1}',
                  f'Train acurracy : {train_accuracy[epoch]}',
                  f'Test accuracy : {test_accuracy[epoch]}')

    if print_plot:
        # Setting x-ticks
        epochs_range = range(1, n_epochs + 1)
        # fig, ax = plt.subplots(nrows=1, ncols=2)
        plt.subplots(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, train_loss, 'g', label='Training loss')
        plt.plot(epochs_range, test_loss, 'b', label='Test loss')
        plt.title('Training and Test Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        # #Ploting both curves, train and val

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, train_accuracy, 'g', label='Training accuracy')
        plt.plot(epochs_range, test_accuracy, 'b', label='Test accuracy')
        plt.title('Training and Test Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()



data = Dataset('diabetes.csv')
n_samples = len(data)
n_test = int(n_samples*0.2)

#train_set, test_set = torch.utils.data.random_split(data,[n_samples-n_test],n_test)
train_set, test_set = torch.utils.data.random_split(data,  [n_samples-n_test, n_test])


train_loader = torch.utils.data.DataLoader(train_set, batch_size = len(train_set), shuffle=True)

test_loader = torch.utils.data.DataLoader(test_set,batch_size = len(test_set),shuffle= False)

learning_rate = 0.01

model = Net(len(data[0][0])).to(DEVICE)

optimizer = optim.Adam(model.parameters(),lr=learning_rate)
loss_fn = nn.BCELoss()

training_loop(
    n_epochs=175,
    optimizer = optimizer,
    model = model,
    loss_fn = loss_fn,
    print_plot=True,
    train_loader = train_loader,
    test_loader=test_loader
)




