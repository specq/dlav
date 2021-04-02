# import the necessary packages
import data_utils as du
import argparse
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import pickle

#########################################################################
# TODO:                                                                 #
# This is used to input our test dataset to your model in order to      #
# calculate your accuracy                                               #
# Note: The input to the function is similar to the output of the method#
# "get_CIFAR10_data" found in the notebooks.                            #
#########################################################################

class ConvNet(nn.Module):
    def __init__(self, n_input_channels=3, n_output=10):
        super().__init__()
        ################################################################################
        # TODO:                                                                        #
        # Define 2 or more different layers of the neural network                      #
        ################################################################################

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(32)
        self.maxPool1 = nn.MaxPool2d(2, stride=2)
        
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.batchNorm3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.batchNorm4 = nn.BatchNorm2d(64)
        self.maxPool2 = nn.MaxPool2d(2, stride=2)
        
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.batchNorm5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.batchNorm6 = nn.BatchNorm2d(128)
        self.maxPool3 = nn.MaxPool2d(2, stride=2)
        
        self.fc1 = nn.Linear(2048, 10)
        
        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################
    
    def forward(self, x):
        ################################################################################
        # TODO:                                                                        #
        # Set up the forward pass that the input data will go through.                 #
        # A good activation function betweent the layers is a ReLu function.           #
        #                                                                              #
        # Note that the output of the last convolution layer should be flattened       #
        # before being inputted to the fully connected layer. We can flatten           #
        # Tensor `x` with `x.view`.                                                    #
        ################################################################################

        x = self.conv1(x)
        x = self.batchNorm1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.batchNorm2(x)
        x = F.relu(x)
        x = self.maxPool1(x)
        
        x = self.conv3(x)
        x = self.batchNorm3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = self.batchNorm4(x)
        x = F.relu(x)
        x = self.maxPool2(x)
        
        x = self.conv5(x)
        x = self.batchNorm5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = self.batchNorm6(x)
        x = F.relu(x)
        x = self.maxPool3(x)
        
        x = self.fc1(x.view(x.size()[0], 2048))
        
        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################
        
        return x
    
    def predict(self, x):
        outputs = self.forward(x)
        return torch.argmax(F.softmax(outputs).data, 1)

def predict_usingCNN(X):
    #########################################################################
    # TODO:                                                                 #
    # - Load your saved model                                               #
    # - Do the operation required to get the predictions                    #
    # - Return predictions in a numpy array                                 #
    #########################################################################

    # Create a net
    net = ConvNet()

    # Load the model
    checkpoint = torch.load('model86.ckpt')
    net.load_state_dict(checkpoint)
    X = torch.from_numpy(X).float()
    net.eval()
    with torch.no_grad():
        y_pred = net.predict(X).numpy()

    #########################################################################
    #                       END OF YOUR CODE                                #
    #########################################################################
    return y_pred


def main(filename, group_number):
    X,Y = du.load_CIFAR_batch(filename)
    mean_pytorch = np.array([0.4914, 0.4822, 0.4465])
    std_pytorch = np.array([0.2023, 0.1994, 0.2010])
    X_pytorch = np.divide(np.subtract( X/255 , mean_pytorch[:,np.newaxis,np.newaxis]), std_pytorch[:,np.newaxis,np.newaxis])
    #import pdb; pdb.set_trace()
    prediction_cnn = predict_usingCNN(X_pytorch)
    acc_cnn = sum(prediction_cnn == Y)/len(X_pytorch)
    print("Group %s ... CNN= %f"%(group_number, acc_cnn))
    
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--test", required=True, help="path to test file")
    ap.add_argument("-g", "--group", required=True, help="group number")
    args = vars(ap.parse_args())
    main(args["test"],args["group"])


