import numpy as np
import sys

def main():
    """""
    main function.
    runs the program.
    """""
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
            for i in range(3):
                optional_max = softmax(i, w, xt, b)  # i its optional tag we want to find the tag with the high probability for xt
                if optional_max > soft_max:
                    soft_max = optional_max
                    y_hat = i
            if (y_hat != y):  # check if y_hat and tag are not equal
                # we need to update w and b
                for i in range(3):
                    if i+1==y:
                        loss_difrenzial_by_w = -xt + np.dot(softmax(y, w, xt, b), xt)
                        loss_difrenzial_by_b = -1 + softmax(y, w, xt, b)
                    else:
                        loss_difrenzial_by_w = np.dot(softmax(y, w, xt, b), xt)
                        loss_difrenzial_by_b = softmax(y, w, xt, b)
                    # update w
                    w = w - np.dot(eta, loss_difrenzial_by_w)
                    # update b
                    b = b - np.dot(eta, loss_difrenzial_by_b)

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



if __name__ == '__main__':
    main()