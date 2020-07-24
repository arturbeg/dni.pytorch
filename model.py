import torch.nn as nn
from dni import *

# Neural Network Model (1 hidden layer)
class mlp(nn.Module):
    def __init__(self, conditioned_DNI, input_size, num_classes, hidden_size=256):
        super(mlp, self).__init__()
        # classify network
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
        # dni network
        self._fc1 = dni_linear(hidden_size, num_classes, conditioned=conditioned_DNI)
        self._fc2 = dni_linear(num_classes, num_classes, conditioned=conditioned_DNI)

        self.mlp = nn.Sequential(self.fc1, self.relu, self.fc2)
        self.dni = nn.Sequential(self._fc1, self._fc2)
        self.optimizers = []
        self.forwards = []
        self.init_optimzers()
        self.init_forwards()
        
    def init_optimzers(self, learning_rate=3e-5):
        self.optimizers.append(torch.optim.Adam(self.fc1.parameters(), lr=learning_rate))
        self.optimizers.append(torch.optim.Adam(self.fc2.parameters(), lr=learning_rate))
        self.optimizer = torch.optim.Adam(self.mlp.parameters(), lr=learning_rate)
        self.grad_optimizer = torch.optim.Adam(self.dni.parameters(), lr=learning_rate)

    def init_forwards(self):
        self.forwards.append(self.forward_fc1)
        self.forwards.append(self.forward_fc2)

    def forward_fc1(self, x, y=None):
        x = x.view(-1, 28*28)
        out = self.fc1(x)
        grad = self._fc1(out, y)
        return out, grad
       
    def forward_fc2(self, x, y=None):
        x = self.relu(x)
        out = self.fc2(x)
        grad = self._fc2(out, y)
        return out, grad
 
    def forward(self, x, calculate_syn_grads=False):
        x = x.view(-1, 28*28)
        fc1 = self.fc1(x)
        relu1 = self.relu(fc1)
        fc2 = self.fc2(relu1)
        
        if calculate_syn_grads:
            grad_fc1 = self._fc1(fc1, y=None)
            grad_fc2 = self._fc2(fc2, y=None)
            return (fc1, fc2), (grad_fc1, grad_fc2)
        else:
            return fc1, fc2
