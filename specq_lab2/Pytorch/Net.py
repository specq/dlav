import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        ################################################################################
        # TODO:                                                                        #
        # Define 2 or more different layers of the neural network                      #
        ################################################################################

        self.fc1 = nn.Linear(3072, 4000)
        self.fc2 = nn.Linear(4000, 1000)
        self.fc3 = nn.Linear(1000,4000)
        self.fc4 = nn.Linear(4000,10)
        
        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################

    def forward(self, x):
        x = x.view(x.size(0),-1)
        ################################################################################
        # TODO:                                                                        #
        # Set up the forward pass that the input data will go through.                 #
        # A good activation function betweent the layers is a ReLu function.           #
        ################################################################################
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)

        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################
        return x

    def predict(self, x):
        outputs = self.forward(x)
        return torch.argmax(F.softmax(outputs).data, 1)