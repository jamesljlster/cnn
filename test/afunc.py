import numpy as np
import math

def softmax(src):
    # Get size of input vector
    rows, cols = src.shape

    # Checking
    if rows > 1:
        raise Exception("Input rows > 1")

    # Find softmax
    expVec = np.exp(src)
    return expVec / np.sum(expVec)

def softmax_derivative(src):
    # Get size of input vector
    rows, cols = src.shape

    # Checking
    if rows > 1:
        raise Exception("Input rows > 1")

    # Find softmax derivative
    tmpVec = softmax(src)
    retMat = np.zeros((cols, cols))
    for i in range(cols):
        for j in range(cols):
            retMat[i, j] = tmpVec[0, i] * (float((i == j)) - tmpVec[0, j])

    return retMat

def relu(src):
    # Get size of input vector
    rows, cols = src.shape

    # Checking
    if rows > 1:
        raise Exception("Input rows > 1")

    # Find relu
    retVec = np.zeros((1, cols))
    for i in range(cols):
        retVec[0, i] = max(src[0, i], 0.0)

    return retVec

def relu_derivative(src):
    # Get size of input vector
    rows, cols = src.shape

    # Checking
    if rows > 1:
        raise Exception("Input rows > 1")

    # Find relu derivative
    retMat = np.zeros((cols, cols))
    for i in range(cols):
        if src[0, i] < 0.0:
            retMat[i, i] = 0
        else:
            retMat[i, i] = 1

    return retMat
