import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.autograd import Variable
from utils import MNIST_loaders

# # Các siêu tham số
# BATCH_SIZE = 500
# N_STEPS = 28
# N_INPUTS = 28
# N_NEURONS = 128
# N_OUTPUTS = 10
# NUM_EPOCHS = 5
# LEARNING_RATE = 0.001

sequence_length = 28
input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 100
num_epochs = 20
learning_rate = 0.01

# # Chuẩn bị dữ liệu
# transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
# train_dataset = MNIST(root='./data', train=True, transform=transform, download=True)
# train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# test_dataset = MNIST(root='./data', train=False, transform=transform, download=True)
# test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# # Định nghĩa mô hình LSTM
# class ImageLSTM(nn.Module):
#     def __init__(self, batch_size, n_steps, n_inputs, n_neurons, n_outputs):
#         super(ImageLSTM, self).__init__()
#         self.n_neurons = n_neurons
#         self.batch_size = batch_size
#         self.n_steps = n_steps
#         self.n_inputs = n_inputs
#         self.n_outputs = n_outputs
#         self.lstm = nn.LSTM(self.n_inputs, self.n_neurons, batch_first=True)
#         self.FC = nn.Linear(self.n_neurons, self.n_outputs)

#     def forward(self, X):
#         X = X.view(-1, self.n_steps, self.n_inputs)
#         self.batch_size = X.size(0)
#         self.hidden, self.cell_state = self.init_hidden()
#         lstm_out, (self.hidden, self.cell_state) = self.lstm(X, (self.hidden, self.cell_state))
#         out = self.FC(self.hidden)
#         return out.view(-1, self.n_outputs)

#     def init_hidden(self):
#         return (torch.zeros(1, self.batch_size, self.n_neurons),
#                 torch.zeros(1, self.batch_size, self.n_neurons))

    
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
optimizer = optim.Adam(model.parameters(), lr = 0.01)   

torch.manual_seed(1234)
train_loader, test_loader = MNIST_loaders(batch_size,batch_size)

# Hàm tính độ chính xác
def get_accuracy(logit, target, batch_size):
    corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
    accuracy = 100.0 * corrects / batch_size
    return accuracy.item()

# Vòng lặp huấn luyện
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
    print ('Epoch [{}/{}],  Loss: {:.4f}' 
                .format(epoch + 1, num_epochs, loss.item()))        
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
