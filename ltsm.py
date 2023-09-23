import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader
from tqdm import tqdm

# Các siêu tham số
BATCH_SIZE = 100
N_STEPS = 28
N_INPUTS = 28
N_NEURONS = 128
N_OUTPUTS = 10
NUM_EPOCHS = 5
LEARNING_RATE = 0.001

# Chuẩn bị dữ liệu
transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
train_dataset = MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataset = MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Định nghĩa mô hình LSTM
class ImageLSTM(nn.Module):
    def __init__(self, batch_size, n_steps, n_inputs, n_neurons, n_outputs):
        super(ImageLSTM, self).__init__()
        self.n_neurons = n_neurons
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.lstm = nn.LSTM(self.n_inputs, self.n_neurons, batch_first=True)
        self.FC = nn.Linear(self.n_neurons, self.n_outputs)

    def forward(self, X):
        X = X.view(-1, self.n_steps, self.n_inputs)
        self.batch_size = X.size(0)
        self.hidden, self.cell_state = self.init_hidden()
        lstm_out, (self.hidden, self.cell_state) = self.lstm(X, (self.hidden, self.cell_state))
        out = self.FC(self.hidden)
        return out.view(-1, self.n_outputs)

    def init_hidden(self):
        return (torch.zeros(1, self.batch_size, self.n_neurons),
                torch.zeros(1, self.batch_size, self.n_neurons))

# Khởi tạo mô hình và bộ tối ưu hóa
model_lstm = ImageLSTM(BATCH_SIZE, N_STEPS, N_INPUTS, N_NEURONS, N_OUTPUTS)
criterion = nn.CrossEntropyLoss()
optimizer_lstm = optim.Adam(model_lstm.parameters(), lr=LEARNING_RATE)

# Hàm tính độ chính xác
def get_accuracy(logit, target, batch_size):
    corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
    accuracy = 100.0 * corrects / batch_size
    return accuracy.item()

# Vòng lặp huấn luyện
for epoch in range(NUM_EPOCHS):
    train_running_loss = 0.0
    train_acc = 0.0
    model_lstm.train()

    for i, data in enumerate(tqdm(train_loader), 0):
        optimizer_lstm.zero_grad()
        model_lstm.hidden, model_lstm.cell_state = model_lstm.init_hidden()
        inputs, labels = data
        inputs = inputs.view(-1, N_STEPS, N_INPUTS)
        outputs = model_lstm(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_lstm.step()
        train_running_loss += loss.detach().item()
        train_acc += get_accuracy(outputs, labels, BATCH_SIZE)

    model_lstm.eval()
    print(f'LSTM - Epoch: {epoch + 1} | Loss: {train_running_loss / i:.4f} | Train Accuracy: {train_acc / i:.2f}')

# Kiểm tra mô hình trên tập kiểm tra
test_acc_lstm = 0.0
model_lstm.eval()
for i, data in enumerate(test_loader, 0):
    inputs, labels = data
    inputs = inputs.view(-1, N_STEPS, N_INPUTS)
    outputs = model_lstm(inputs)
    test_acc_lstm += get_accuracy(outputs, labels, BATCH_SIZE)

print(f'LSTM - Test Accuracy: {test_acc_lstm / i:.2f}')
