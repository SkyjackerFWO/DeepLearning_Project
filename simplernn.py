import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from utils import MNIST_loaders, save_model  # Make sure 'utils.py' is available with these functions

# Create a directory to store the TensorBoard logs and model weights
if not os.path.exists('runs'):
    os.makedirs('runs')
if not os.path.exists('weights'):
    os.makedirs('weights')

writer = SummaryWriter('runs/rnn_mnist')

save_path = 'weights/rnn_mnist'
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
model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

torch.manual_seed(1234)
train_loader, test_loader = MNIST_loaders(batch_size, batch_size)

# Function to calculate accuracy
def get_accuracy(logit, target, batch_size):
    corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
    accuracy = 100.0 * corrects / batch_size
    return accuracy.item()

# Visualize images in TensorBoard
images, labels = next(iter(train_loader))
grid = torchvision.utils.make_grid(images)
writer.add_image('images', grid)
images = images.reshape(-1, sequence_length, input_size).to(device)
writer.add_graph(model, images)

# Training loop
running_loss = 0
best_loss = 0
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
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    writer.add_scalar('training loss', avg_loss, epoch)

    # Calculate accuracy
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

        accuracy = 100 * correct / total
        writer.add_scalar('accuracy', accuracy, epoch)

    print('Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%, Best loss: {:.4f}'.format(epoch + 1, num_epochs, avg_loss, accuracy, best_loss))
    best_loss = save_model(model, optimizer, epoch, best_loss, avg_loss, save_path)
    running_loss = 0.0

# Close the TensorBoard writer
writer.close()
