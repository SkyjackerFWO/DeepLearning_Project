import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import MNIST_loaders, save_model
writer = SummaryWriter('runs/cnn_mnist')

save_path = 'weights/cnn_mnist'
sequence_length = 28
input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 100
num_epochs = 50
learning_rate = 0.001


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

##################################################################################################################

# Khởi tạo mô hình và bộ tối ưu hóa
model = CNN(num_classes).to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)   

torch.manual_seed(1234)
train_loader, test_loader = MNIST_loaders(batch_size,batch_size)


# Hàm tính độ chính xác
def get_accuracy(logit, target, batch_size):
    corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
    accuracy = 100.0 * corrects / batch_size
    return accuracy.item()

# Vòng lặp huấn luyện
running_loss = 0
best_loss = 0

for epoch in range(num_epochs):
    total_step = len(train_loader)
    model.train()

    for i, (images, labels) in enumerate(tqdm(train_loader)):
        
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = F.nll_loss(output, labels) 
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    writer.add_scalar('training loss',
                    avg_loss,
                    epoch)
    best_loss = save_model(model, optimizer, epoch, best_loss, avg_loss, save_path)
    print ('Epoch [{}/{}],  Loss: {:.4f}, Best loss: {:.4f}' 
                .format(epoch + 1, num_epochs, avg_loss, best_loss))
    running_loss = 0.0        
# Test the model
# model.eval()
# with torch.no_grad():
#     test_loss = 0
#     correct = 0
#     for data, target in test_loader:
#         data, target = Variable(data, volatile=True), Variable(torch.squeeze(target))
#         output = model(data)
#         test_loss += F.nll_loss(output, target).data[0]
#         pred = output.data.max(1)[1] # get the index of the max log-probability
#         correct += pred.eq(target.data).cpu().sum()

#     test_loss = test_loss
#     test_loss /= len(test_loader) # loss function already averages over batch size
#     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         test_loss, correct, len(test_loader.dataset),
#         100. * correct / len(test_loader.dataset)))
# # print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

model.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

test_loss /= len(test_loader.dataset)

print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))
