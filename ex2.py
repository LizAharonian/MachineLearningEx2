import numpy as np
import sys
import matplotlib.pyplot as plt
import math

def main():
    """""
    main function.
    runs the program.
    """""
    check()
    w = [0,0,0]
    b = [0,0,0]
    eta = 1 #mekadem lemmida
    #set of examples and tags
    s ={1:create_sampels(1),2:create_sampels(2),3:create_sampels(3)}
    for y in s.keys():

        sampels = s[y]

        for xt in sampels:
            soft_max = 0
            y_hat = 0
            for i in range(1,4):
                optional_max = softmax(i, w, xt, b)  # i its optional tag we want to find the tag with the high probability for xt
                if optional_max > soft_max:
                    soft_max = optional_max
                    y_hat = i
           # if (y_hat != y):  # check if y_hat and tag are not equal
                # we need to update w and b
            for i in range(1,4):
                if i==y:
                    loss_difrenzial_by_w = -xt + np.dot(softmax(y, w, xt, b), xt)
                    loss_difrenzial_by_b = -1 + softmax(y, w, xt, b)
                else:
                    loss_difrenzial_by_w = np.dot(softmax(y, w, xt, b), xt)
                    loss_difrenzial_by_b = softmax(y, w, xt, b)
                # update w
                w[i-1] = w[i-1] - np.dot(eta, loss_difrenzial_by_w)
                # update b
                b[i-1] = b[i-1] - np.dot(eta, loss_difrenzial_by_b)

    print w
    print b
    print "liz"



def create_sampels(a):
    return np.random.normal(2 * a, 1, 100)

def softmax(a,w,xt,b):
    # calculate the sum
    sum = 0
    for j in range(3):
        sum += np.exp(np.dot(w[j], xt) + b[j])
    return np.divide(np.exp(np.dot(w[a-1],xt)+b[a-1]),sum)

def get_y_hat(w, b,xt):
    for i in range(1, 4):
        optional_max = softmax(i, w, xt,  b)  # i its optional tag we want to find the tag with the high probability for xt
        if optional_max > soft_max:
            soft_max = optional_max
            y_hat = i
    return y_hat

def check():
    dictSoftmax ={}
    w = [-364.36063416573586, -188.73461538225644, 22.643354701415333]
    b = [-74.63379423952895, -75.81174030989155, -77.23062127132718]
    for xt in range(1,11):
        #todo:what the hell am i supposed to do here??!
        dictSoftmax[xt] = softmax(1,w,xt ,b)/(softmax(1, w,xt,b) +softmax(2, w,xt,b) +softmax(3, w,xt,b))
    dictReal ={}
    for xt in range(1,11):
        dictReal[xt] = (density(2, xt)) / (density(2, xt) + density(4, xt) + density(6, xt))
    plt.plot(dictSoftmax.keys(), dictSoftmax.values(), "b-", label='Real')
    plt.plot(dictReal.keys(), dictReal.values(), "r-", label='Real')
    plt.show()


    print "lala"

def density(m, xt):
    return ((1/math.sqrt(2*math.pi)) * np.exp((-(xt-m)**2)/2))


if __name__ == '__main__':
    main()