import torch
from torch import nn
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class layer(nn.Module):
    def __init__(self, parameters, outputs, bias = True):
        super().__init__()
        self.parameters = parameters
        self.outputs = outputs
        self.input_in = parameters/2
        self.output_out = outputs/2
        self.linear = nn.Linear(self.input_in, self.output_out, bias)
        boundary = (6 ** 0.5 / (self.input_in + self.output_out) ** 0.5) 
        nn.init.uniform_(self.linear.weight, -boundary, boundary)
        
    def forward(self, x):
        size = x.shape[0]
        splitBatch = torch.split(x, self.input_in, dim=1)
        print("Second Step:")
        print(splitBatch)
        InStackBatch = torch.cat(splitBatch, dim = 0)
        OutStackBatch = F.relu(self.linear(InStackBatch))
        OutBatch = torch.split(OutStackBatch, size, dim=0) 
        print("Third Step:")
        print(OutBatch)
        x = torch.cat(OutBatch, dim = 1)
        print("Last (Fourth) step:")
        return x
    
class DropNorm(nn.Module):
    def __init__(self, num_features, dropout_prob=0.5, eps = 1e-6):
        super(DropNorm, self).__init__()
        self.num_features = num_features
        self.dropout_prob = dropout_prob
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.dropout = nn.Dropout(dropout_prob)
        self.eps = eps

    def forward(self, x):
        # Calculate mean and variance for each feature across the batch
        mean = x.mean(dim=0, keepdim=True)
        var = x.var(dim=0, unbiased=False, keepdim=True)
        x_normalized = (x - mean) / torch.sqrt(var + self.eps) # normalization
        
        mask = (torch.rand_like(x_normalized) > self.dropout_prob).float() # mask for non-dropped out neurons
        x_dropped = x_normalized * mask
        
        # rescaling
        out = self.gamma * x_dropped + self.beta
        
        return out

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.drop_norm = DropNorm(256, 0.5)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 784)  # Flatten the input
        x = F.relu(self.fc1(x))
        x = self.drop_norm(x)
        x = self.fc2(x)
        return x

class DropoutNet(nn.Module):
    def __init__(self):
        super(DropoutNet, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 784)  # Flatten the input
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def test_model_accuracy(model, train_loader, test_loader):
    criterion = nn.CrossEntropyLoss() # cross entropy loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # adam optimizer algorithm
    
    # Training loop
    for epoch in range(5):
        model.train()
        for data, labels in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
    
    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy
    
input = torch.rand(4,4)
print("First Step:")
print(input)
Example = layer(4,4)
print(Example.forward(input))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # use GPU
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Load the Fashion-MNIST dataset
train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transform, download = True)
test_dataset = datasets.FashionMNIST(root='./data', train=False, transform=transform, download = True)
train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Create instances of both models
dropout_model = DropoutNet().to(device)
dropnorm_model = SimpleNet().to(device)

# Test accuracy of both models
dropout_accuracy = test_model_accuracy(dropout_model, train_loader, test_loader)
dropnorm_accuracy = test_model_accuracy(dropnorm_model, train_loader, test_loader)

print(f"Dropout model accuracy: {dropout_accuracy:.2f}%")
print(f"DropNorm model accuracy: {dropnorm_accuracy:.2f}%")
