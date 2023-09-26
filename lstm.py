import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader
from tqdm import tqdm
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
num_epochs = 2
learning_rate = 0.01

# # Chuẩn bị dữ liệu
# transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
# train_dataset = MNIST(root='./data', train=True, transform=transform, download=True)
# train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# test_dataset = MNIST(root='./data', train=False, transform=transform, download=True)
# test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


class ImageLSTM(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(ImageLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        pass
    
    def forward(self, x):
        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # Passing in the input and hidden state into the model and  obtaining outputs
        out, hidden = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        #Reshaping the outputs such that it can be fit into the fully connected layer
        out = self.fc(out[:, -1, :])
        return out
    
class RNN(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        pass
    
    def forward(self, x):
        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # Passing in the input and hidden state into the model and  obtaining outputs
        out, hidden = self.rnn(x, h0)  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        #Reshaping the outputs such that it can be fit into the fully connected layer
        out = self.fc(out[:, -1, :])
        return out

##################################################################################################################

# Khởi tạo mô hình và bộ tối ưu hóa
model = ImageLSTM(input_size, hidden_size, num_layers, num_classes).to(device)
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

    for i, (images, labels) in enumerate(tqdm(train_loader), 0):
        
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        # Forward pass
        outputs = model(images)
        loss = loss_func(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # train_running_loss += loss.detach().item()
        # train_acc += get_accuracy(outputs, labels, batch_size)

    
    print ('Epoch [{}/{}],  Loss: {:.4f}' 
                .format(epoch + 1, num_epochs, loss.item()))

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total = total + labels.size(0)
        correct = correct + (predicted == labels).sum().item()
print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))


