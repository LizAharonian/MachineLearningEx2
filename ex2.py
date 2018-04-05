import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D

import math

def main():
    """""
    main function.
    runs the program.
    """""
    # initialize w and b
    w = [0,0,0]
    b = [0,0,0]
    # learning rate
    eta = 0.3
    # set of examples and tags
    s =[]
    for tag in range(1,4):
        sampels = create_sampels(tag)
        for i in range(100):
            s.append((tag,sampels[i]))
    np.random.shuffle(s)
    # train the algorithm
    (w,b) = training(s,eta,w,b)
    # test the algorithm
    check(w,b)


def training(s,eta,w,b):
    """""
    training function.
    updates w and b using loss function and softmax.
    """""
    epochs = 10
    for e in range(epochs):
        for (y, xt) in s:
            # we need to update w and b
            for i in range(1, 4):
                if i == y:
                    loss_difrenzial_by_w = -xt + softmax(i, w, xt, b) * xt
                    loss_difrenzial_by_b = -1 + softmax(i, w, xt, b)
                else:
                    loss_difrenzial_by_w = softmax(i, w, xt, b) * xt
                    loss_difrenzial_by_b = softmax(i, w, xt, b)
                # update w
                w[i - 1] = w[i - 1] - eta * loss_difrenzial_by_w
                # update b
                b[i - 1] = b[i - 1] - eta * loss_difrenzial_by_b
    return (w,b)

def create_sampels(a):
    """""
    create_sampels function.
    creates 100 random examples of a normal dist.
    """""
    return np.random.normal(2 * a, 1.0, 100)

def softmax(a,w,xt,b):
    """""
    softmax function.
    clculates the probability xt tag is a.
    """""
    # calculate the sum
    sum = 0
    for j in range(3):
        sum += np.exp(w[j] * xt + b[j])
    return (np.exp(w[a-1]*xt+b[a-1]))/sum

def check(w,b):
    """""
    check function.
    creates 10 examples and plots their probability to be belong to 1 normal dist.
    """""
    dictSoftmax ={}
    for xt in range(0,10):
        dictSoftmax[xt] = softmax(1,w,xt,b)
    dictReal ={}
    for xt in range(0,10):
        dictReal[xt] = (density(2, xt)) / (density(2, xt) + density(4, xt) + density(6, xt))
    label1, =plt.plot(dictSoftmax.keys(), dictSoftmax.values(), "b-", label='Softmax Distribution')
    label2, =plt.plot(dictReal.keys(), dictReal.values(), "r-", label='Real normal Distribution')
    plt.legend(handler_map = {label1:HandlerLine2D(numpoints=4)})
    plt.show()


def density(m, xt):
    """""
    density function.
    returns normal density function with expectancy m.
    """""
    return ((1.0/math.sqrt(2*math.pi)) * np.exp((-(xt-m)**2)/2))


if __name__ == '__main__':
    main()