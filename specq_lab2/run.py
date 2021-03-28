# import the necessary packages
import Softmax.data_utils as du
import argparse
import numpy as np
import torch
from Softmax.linear_classifier import Softmax
from Pytorch.Net import Net
import pickle

#########################################################################
# TODO:                                                                 #
# This is used to input our test dataset to your model in order to      #
# calculate your accuracy                                               #
# Note: The input to the function is similar to the output of the method#
# "get_CIFAR10_data" found in the notebooks.                            #
#########################################################################

def predict_usingPytorch(X):
    #########################################################################
    # TODO:                                                                 #
    # - Load your saved model                                               #
    # - Do the operation required to get the predictions                    #
    # - Return predictions in a numpy array                                 #
    #########################################################################

    # Create a net
    net = Net()

    # Load the model
    checkpoint = torch.load('Pytorch/best_model.ckpt')
    net.load_state_dict(checkpoint)

    y_pred = net.predict(X).numpy()

    #########################################################################
    #                       END OF YOUR CODE                                #
    #########################################################################
    return y_pred

def predict_usingSoftmax(X):
    #########################################################################
    # TODO:                                                                 #
    # - Load your saved model                                               #
    # - Do the operation required to get the predictions                    #
    # - Return predictions in a numpy array                                 #
    #########################################################################

    # Open the weights
    with open('Softmax/softmax_weights.pkl', 'rb') as f:
        W = pickle.load(f)

    # Create a classifier
    softmax_obj = Softmax()

    # Copy the weights
    softmax_obj.W = W.copy()

    # Predict the classes
    y_pred = softmax_obj.predict(X)
    #########################################################################
    #                       END OF YOUR CODE                                #
    #########################################################################
    return y_pred

def main(filename, group_number):

    X,Y = du.load_CIFAR_batch(filename)
    ### Modified this part
    mean_pytorch = np.array([0.4914, 0.4822, 0.4465])
    std_pytorch = np.array([0.2023, 0.1994, 0.2010])
    X_pytorch = np.divide(np.subtract( X/255 , mean_pytorch[np.newaxis,np.newaxis,:]), std_pytorch[np.newaxis,np.newaxis,:])
    prediction_pytorch = predict_usingPytorch(torch.Tensor(np.moveaxis(X_pytorch,-1,1)))
    ####
    X = np.reshape(X, (X.shape[0], -1))
    mean_image = np.mean(X, axis = 0)
    X -= mean_image
    X = np.hstack([X, np.ones((X.shape[0], 1))])
    prediction_softmax = predict_usingSoftmax(X)
    acc_softmax = sum(prediction_softmax == Y)/len(X)
    acc_pytorch = sum(prediction_pytorch == Y)/len(X)
    print("Group %s ... Softmax= %f ... Pytorch= %f"%(group_number, acc_softmax, acc_pytorch))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--test", required=True, help="path to test file")
    ap.add_argument("-g", "--group", required=True, help="group number")
    args = vars(ap.parse_args())
    main(args["test"],args["group"])



