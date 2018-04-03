import numpy as np
import sys

def main():
    """""
    main function.
    runs the program.
    """""
    sampels_1 = create_sampels(1)
    w = [0,0,0]
    b = [0,0,0]
    n = 1 #mekadem lemmida
    for xt in sampels_1:
        soft_max =0
        y_hat = 0
        sum = 0
        # calculat the sum
        for j in range(3):
            sum += np.exp(np.dot(w[j],xt)+b[j])

        for i in range(3):

            optional_max = np.exp(np.dot(w[i],xt)+b[i])
            if optional_max>soft_max:
                soft_max = optional_max
                y_hat =i
        #if (y_hat !=1):#check if y_hat and tag are not equal
            #we need to update w and b







    print "liz"



def create_sampels(a):
    return np.random.normal(2 * a, 1, 100)


if __name__ == '__main__':
    main()