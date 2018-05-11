import numpy as np
import pickle
sigmoid = lambda x: 1 / (1 + np.exp(-x))

def main():


    train_x = np.loadtxt("train_x")
    train_y = np.loadtxt("train_y")
    test_x = np.loadtxt("test_x")
    print "collected"

    # shuffle the training set
    shape = np.arange(train_x.shape[0])
    np.random.shuffle(shape)
    train_x = train_x[shape]
    train_y = train_y[shape]

    #split to val_set and train_set
    val_size = int(len(train_x) *0.2)
    val_x = train_x[-val_size:, :]
    val_y = train_y[-val_size:]
    train_x = train_x[: -val_size, :]
    train_y = train_y[: -val_size]
    #help params
    prob_dimend = 784 #num of picksels in pic

    #hyper paramemters
    H = 20   #size of hidden layer
    epochs = 20
    eta = 0.1

    W1 = np.random.rand(H, prob_dimend)
    b1 = np.random.rand(H, 1)
    W2 = np.random.rand(10, H)
    b2 = np.random.rand(10, 1)
    params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

    #train the model
    train(params,epochs,sigmoid,eta,train_x,train_y,val_x,val_y)

    x = np.random.rand(2, 1)
    y = np.random.randint(0, 2)  # Returns 0/1

    fprop_cache = fprop(x, y, params)
    bprop_cache = bprop(fprop_cache)

    # Numerical gradient checking
    # Note how slow this is! Thus we want to use the backpropagation algorithm instead.
    eps = 1e-6
    ng_cache = {}
    # For every single parameter (W, b)
    for key in params:
        param = params[key]
        # This will be our numerical gradient
        ng = np.zeros(param.shape)
        for j in range(ng.shape[0]):
            for k in range(ng.shape[1]):
                # For every element of parameter matrix, compute gradient of loss wrt
                # that element numerically using finite differences
                add_eps = np.copy(param)
                min_eps = np.copy(param)
                add_eps[j, k] += eps
                min_eps[j, k] -= eps
                add_params = np.copy(params)
                min_params = np.copy(params)
                add_params[key] = add_eps
                min_params[key] = min_eps
                ng[j, k] = (fprop(x, y, add_params)['loss'] - fprop(x, y, min_params)['loss']) / (2 * eps)
        ng_cache[key] = ng

    # Compare numerical gradients to those computed using backpropagation algorithm
    for key in params:
        print(key)
        # These should be the same
        print(bprop_cache[key])
        print(ng_cache[key])

    print "liz"

def shuffle(x_arr,y_arr):
    shape = np.arange(x_arr.shape[0])
    np.random.shuffle(shape)
    train_x = x_arr[shape]
    train_y = y_arr[shape]

def train(params,epochs,active_func,eta,train_x, train_y, val_x,val_y):
    for i in xrange(epochs):
        shuffle(train_x, train_y)

        print "train"

if __name__ == '__main__':
    main()



def fprop(x, y, params):
  # Follows procedure given in notes
  W1, b1, W2, b2 = [params[key] for key in ('W1', 'b1', 'W2', 'b2')]
  z1 = np.dot(W1, x) + b1
  h1 = sigmoid(z1)
  z2 = np.dot(W2, h1) + b2
  h2 = sigmoid(z2)
  loss = -(y * np.log(h2) + (1-y) * np.log(1-h2))
  ret = {'x': x, 'y': y, 'z1': z1, 'h1': h1, 'z2': z2, 'h2': h2, 'loss': loss}
  for key in params:
    ret[key] = params[key]
  return ret

def bprop(fprop_cache):
  # Follows procedure given in notes
  x, y, z1, h1, z2, h2, loss = [fprop_cache[key] for key in ('x', 'y', 'z1', 'h1', 'z2', 'h2', 'loss')]
  dz2 = (h2 - y)                                #  dL/dz2
  dW2 = np.dot(dz2, h1.T)                       #  dL/dz2 * dz2/dw2
  db2 = dz2                                     #  dL/dz2 * dz2/db2
  dz1 = np.dot(fprop_cache['W2'].T,
    (h2 - y)) * sigmoid(z1) * (1-sigmoid(z1))   #  dL/dz2 * dz2/dh1 * dh1/dz1
  dW1 = np.dot(dz1, x.T)                        #  dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/dw1
  db1 = dz1                                     #  dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/db1
  return {'b1': db1, 'W1': dW1, 'b2': db2, 'W2': dW2}

