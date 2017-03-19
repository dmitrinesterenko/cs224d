import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck

def forward_backward_prop(data, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    ### YOUR CODE HERE: forward propagation
    hidden = sigmoid(np.dot(data, W1) + b1)
    scores = softmax(np.dot(hidden, W2) + b2)
    probabilities = -np.log(scores)
    # loss is found by comparing scores against labels
    y = np.argmax(labels, axis=1)
    dscores = probabilities
    dscores[range(data.shape[0]), y] -= 1
    dscores /= data.shape[0]

    ### END YOUR CODE

    ### YOUR CODE HERE: backward propagation
    gradW2 = np.dot(hidden.T, dscores)
    gradb2 = np.sum(dscores, axis=0, keepdims=True)
    hidden_derivative = np.dot(dscores, W2.T)
    gradW1 = np.dot(data.T, hidden_derivative)
    gradb1 = np.sum(hidden_derivative, axis=0, keepdims=True)
    ### END YOUR CODE

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))

    return probabilities, grad

def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print "Running sanity check..."

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i,random.randint(0,dimensions[2]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    import pdb; pdb.set_trace()

    gradcheck(lambda params: forward_backward_prop(data, labels, params,
        dimensions), params)

def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    ### YOUR CODE HERE
    print "I am feeling lucky!"
    ### END YOUR CODE

if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()
