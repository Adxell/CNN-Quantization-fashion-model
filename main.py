import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F


from torch.utils.tenserboard import SummaryWriter
from datetime import datetime

from copy import deepcopy


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
validation_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)


train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=4, shuffle=True)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=4, shuffle=False)

classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
model = SimpleCNN()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

now = datetime.now().strftime('%b%d_%H-%M-%S')

def train_one_epoch(epoch, tb_writer):
    running_loss = 0.0
    last_loss = 0.0

    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000
            print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {last_loss:.3f}')
            tb_x = epoch * len(train_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.0
    
    return last_loss

writer = SummaryWriter(f'runs/fashion_mnist_experiment_{now}')

epochs_number = 0


EPOCHS = 5

best_vloss = 1_000_000.0


for epoch in range(EPOCHS):
    print(f'Starting epoch {epoch + 1}...')

    model.train(True)
    avg_loss = train_one_epoch(epoch, writer)

    running_vloss = 0.0

    model.eval()

    with torch.no_grad():
        for data in validation_loader:
            inputs, labels = data
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_vloss += loss.item()

    avg_vloss = running_vloss / len(validation_loader)
    print(f'Epoch {epoch + 1} completed. Last loss: {avg_loss:.3f}, Validation loss: {avg_vloss:.3f}')

    writer.add_scalars('Training vs. Validation Loss', {'Training': avg_loss, 'Validation': avg_vloss}, epochs_number + 1)
    writer.flush()

    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'model_{}_{}'.format(now, epoch)
        print(f'Validation loss decreased ({best_vloss:.3f} --> {avg_vloss:.3f}).  Saving model ...')
        torch.save(model.state_dict(), model_path)
    epochs_number += 1


def show_param_dtype(model):
    for param in model.parameters():
        print("Type:", param.dtype, "Shape:", param.shape)

print("Model's state_dict in float 32:")
show_param_dtype(model)

model_fp16 = deepcopy(model)

model_fp16 = model_fp16.to(torch.float16)

print("Model's state_dict in float 16:")
show_param_dtype(model_fp16)


def evaluate_model(model):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in validation_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy