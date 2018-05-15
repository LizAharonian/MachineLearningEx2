import numpy as np


def main():
    """""
    main function.
    runs the program.
    implement of NN.
    
    """""

    #train_x = np.loadtxt("train_x")
    #train_y = np.loadtxt("train_y")
    #test_x = np.loadtxt("test_x")

#*******************************************
    #load set of examples
    #todo:change back to normal reading
    train_x = np.load("train_x.bin.npy")
    #np.save("train_x.bin",train_x)
    train_y = np.load("train_y.bin.npy")
    #np.save("train_y.bin",train_y)

    test_x = np.load("test_x.bin.npy")
    #np.save("test_x.bin",test_x)

    print "collected"
# *******************************************

    #shuffle the training set
    (train_x,train_y) = shuffle(train_x,train_y)

    #split to val_set and train_set
    val_size = int(len(train_x) *0.2)
    val_x = train_x[-val_size:, :]
    val_y = train_y[-val_size:]
    train_x = train_x[: -val_size, :]
    train_y = train_y[: -val_size]
    #normalization
    train_x=train_x/255
    val_x = val_x/255
    test_x=test_x/255

    #help params
    prob_dimend = 784   #num of picksels in pic

    #hyper paramemters
    H = 100             #size of hidden layer
    epochs = 50
    eta = 0.005

    #initialize params
    W1 = np.random.uniform(-0.08,0.08,[H, prob_dimend])
    b1 = np.random.uniform(-0.08,0.08,[H,1])
    W2 = np.random.uniform(-0.08,0.08,[10, H])
    b2 = np.random.uniform(-0.08,0.08,[10,1])
    params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

    #train the model + validation
    params = train(params,epochs,sigmoid,eta,train_x,train_y,val_x,val_y)

    #save test.pred
    pred_file = open("test.pred", 'w')
    for x in test_x:
        x = np.reshape(x, (1, prob_dimend))
        (fprop_cache, params) = fprop(params, sigmoid, x,0)
        y_hat = fprop_cache['softmax'].argmax(axis=0)
        pred_file.write(str(y_hat[0]) + "\n")
    pred_file.close()


def shuffle(x_arr,y_arr):
    """""
    shuffle function.
    shuffles the arrays togther
    
    """""
    shape = np.arange(x_arr.shape[0])
    np.random.shuffle(shape)
    x_arr = x_arr[shape]
    y_arr = y_arr[shape]
    return (x_arr,y_arr)

def sigmoid(x):
    """""
    sigmoid function.
    convert x to number between 0 and 1

    """""
    return np.divide(1, (1 + np.exp(-x)))

def loss_func(y_prob):
    """""
    loss_func function.
    calculates the loss

    """""
    return -np.log(y_prob)

def validation(params, active_func, val_x,val_y):
    """""
    validation function.
    runs on validation arrays.
    update rull is not performed here.

    """""
    sum_loss =0
    num_of_success = 0
    for x,y in zip(val_x,val_y):
        x = np.reshape(x, (1, 784))
        (fprop_cache, params) = fprop(params, active_func, x, y)
        y_prob = (fprop_cache['softmax'])[int(y)][0]
        loss = loss_func(y_prob)
        sum_loss += loss
        y_hat = fprop_cache['softmax'].argmax(axis=0)
        if (y == y_hat[0]):
            num_of_success+=1
    accurate = num_of_success / float(np.shape(val_x)[0])
    average_loss = sum_loss /np.shape(val_x)[0]
    return average_loss,accurate

def train(params,epochs,active_func,eta,train_x, train_y, val_x,val_y):
    """""
    train function.
    trains our model and update the params.
    in addition i perform validation after each epoch

    """""
    for i in xrange(epochs):
        sum_loss = 0
        (train_x, train_y)=shuffle(train_x, train_y)
        for x,y in zip(train_x,train_y):
            x = np.reshape(x,(1,784))
            (fprop_cache,params) = fprop(params,active_func,x,y)
            y_prob = (fprop_cache['softmax'])[int(y)][0]
            loss = loss_func(y_prob)
            sum_loss+=loss
            bprop_cache = bprop(fprop_cache)
            params = update_params(params, eta,bprop_cache)

        #perform validation
        val_loss, accurate = validation(params,active_func,val_x,val_y)
        print i , sum_loss/np.shape(train_x)[0], val_loss,accurate*100
    return params


def update_params(params, eta,bprop_cache):
    """""
    update_params function.
    update the params by GD update rule

    """""
    W1, b1, b2, W2 = [params[key] for key in ('W1', 'b1', 'b2','W2')]
    db1, dW1, db2, dW2 = [bprop_cache[key] for key in ('db1', 'dW1', 'db2', 'dW2')]
    W1 = W1 -eta *dW1
    W2 = W2 -eta*dW2
    b2 =b2 -eta*db2
    b1 = b1 -eta*db1
    ret = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    return ret


def softmax(w,xt,b):
    """""
    softmax function.
    calculates the probability that xt's tag is a.
    """""
    # calculate the sum
    sum = 0
    for j in range(10):
        sum += np.exp(np.dot(w[j], xt) + b[j])

    softmax_vec =np.zeros((10,1))
    for i in range(10):
        softmax_vec[i]=(np.exp(np.dot(w[i],xt)+b[i]))/sum

    return softmax_vec

def fprop(params,active_function,x,y):
    """""
    fprop function.
    calculates the NN relevant params

    """""
    # Follows procedure given in notes
    W1, b1, W2, b2 = [params[key] for key in ('W1', 'b1', 'W2', 'b2')]
    x = np.transpose(x)
    z1 = np.dot(W1, x) + b1
    h1 = active_function(z1)
    z2 = np.dot(W2, h1) + b2
    ret = {'x': x, 'y': y, 'z1': z1, 'h1': h1, 'z2': z2, 'softmax': softmax(W2,h1,b2)}
    for key in params:
        ret[key] = params[key]
    return (ret,params)


def bprop(fprop_cache):
    """""
    bprop function.
    calculates the gradients
  
    """""
    # Follows procedure given in notes
    x, y, z1, h1, z2, softmax = [fprop_cache[key] for key in ('x', 'y', 'z1', 'h1', 'z2', 'softmax')]
    y_vec = np.zeros((10,1))
    y_vec[int(y)] =1
    y = y_vec
    dz2 = (softmax - y)                                #  dL/dz2
    dW2 = np.dot(dz2, h1.T)                       #  dL/dz2 * dz2/dw2
    db2 = dz2                                     #  dL/dz2 * dz2/db2
    dz1 = np.dot(fprop_cache['W2'].T,
               (softmax - y)) * sigmoid(z1) * (1 - sigmoid(z1))  # dL/dz2 * dz2/dh1 * dh1/dz1
    dW1 = np.dot(dz1, x.T)                        # dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/dw1
    db1 = dz1                                     #  dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/db1
    return {'db1': db1, 'dW1': dW1, 'db2': db2, 'dW2': dW2}

if __name__ == '__main__':
    main()
