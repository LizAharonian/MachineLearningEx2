import numpy as np
import sys
import matplotlib.pyplot as plt
import math

def main():
    """""
    main function.
    runs the program.
    """""
    w = [0,0,0]
    b = [0,0,0]
    eta = 0.3 #learning rate
    #set of examples and tags
    s =[]
    for tag in range(1,4):
        sampels = create_sampels(tag)
        for i in range(100):
            s.append((tag,sampels[i]))
    np.random.shuffle(s)

    epochs = 10
    for e in range(epochs):
        for (y,xt) in s:

                # we need to update w and b
            for i in range(1,4):
                if i==y:
                    loss_difrenzial_by_w = -xt + softmax(i, w, xt, b)* xt
                    loss_difrenzial_by_b = -1 + softmax(i, w, xt, b)
                else:
                    loss_difrenzial_by_w = softmax(i, w, xt, b)* xt
                    loss_difrenzial_by_b = softmax(i, w, xt, b)
                # update w
                w[i-1] = w[i-1] - eta* loss_difrenzial_by_w
                # update b
                b[i-1] = b[i-1] -eta* loss_difrenzial_by_b
    check(w,b)

    print w
    print b
    print "liz"



def create_sampels(a):
    return np.random.normal(2 * a, 1.0, 100)

def softmax(a,w,xt,b):
    # calculate the sum
    sum = 0
    for j in range(3):
        sum += np.exp(w[j] * xt + b[j])
    return (np.exp(w[a-1]*xt+b[a-1]))/sum

def get_y_hat(w, b,xt):
    soft_max=0
    y_hat=1
    for i in range(1, 4):
        optional_max = softmax(i, w, xt,  b)  # i its optional tag we want to find the tag with the high probability for xt
        if optional_max > soft_max:
            soft_max = optional_max
            y_hat = i
    return (y_hat,soft_max)

def check(w,b):
    dictSoftmax ={}
    for xt in range(0,10):
        dictSoftmax[xt] = softmax(1,w,xt,b)#/(softmax(1,w,xt,b)+softmax(2,w,xt,b)+softmax(3,w,xt,b))
        print str(xt) +" "+str(get_y_hat(w,b,xt))
    dictReal ={}
    for xt in range(0,10):
        dictReal[xt] = (density(2, xt)) / (density(2, xt) + density(4, xt) + density(6, xt))
    plt.plot(dictSoftmax.keys(), dictSoftmax.values(), "b-", label='Real')
    plt.plot(dictReal.keys(), dictReal.values(), "r-", label='Real')
    plt.show()


    print "lala"

def density(m, xt):
    return ((1.0/math.sqrt(2*math.pi)) * np.exp((-(xt-m)**2)/2))


if __name__ == '__main__':
    main()