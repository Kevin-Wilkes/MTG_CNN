import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import pickle


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 20
batch_size = 32
learning_rate = 0.001
X = []
Y = []



pickle_in = open("X2.pickle","rb")
X = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open("Y2.pickle","rb")
Y = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open("Cat2.pickle","rb")
classes = pickle.load(pickle_in)
pickle_in.close()


label_amount = len(classes)



#class cardDataset():
#    def __init__(self, X, Y):
#        self.x = X
#        self.y = Y
#        self.n_samples = len(self.x)

#    def __getitem__(self, index):
#        return self.x[index], self.y[index]

#    def __len__(self):
#        return self.n_samples

class ConvNet(torch.nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(64 * 25 * 35, 256)
        self.fc2 = nn.Linear(256, num_classes)

        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x



images_tensor = torch.tensor(X)
X = []
labels_tensor = torch.tensor(Y)
Y = []


images_tensor = images_tensor.permute(0, 3, 1, 2)
images_tensor = images_tensor.float()



images_tensor /= 255.0


dataset = TensorDataset(images_tensor, labels_tensor)


train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size


train_dataset, test_dataset = random_split(dataset, [train_size, test_size])


train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

model = ConvNet(num_classes = label_amount).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
n_total_steps = len(train_loader)


for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if(i+1) % int(n_total_steps/4) == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

print('Finished Training!')


PATH = './cnn.pth'
torch.save(model.state_dict(), PATH)


with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(label_amount)]
    n_class_samples = [0 for i in range(label_amount)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        
        for i in range(len(labels)):
            label = labels[i].item()
            pred = predicted[i].item()
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    for i in range(label_amount):
        if n_class_samples[i] != 0:
            acc = 100.0 * n_class_correct[i] / n_class_samples[i]
            print(f'Accuracy of {classes[i]}: {acc} %')
