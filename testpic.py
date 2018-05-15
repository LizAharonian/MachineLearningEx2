
import numpy as np
import pickle
import matplotlib.pyplot as plt


test_x = np.loadtxt("test_x")

for im in test_x:
    # M = test_x[0]

    image = np.array(im, dtype=np.uint8)  # [...,::-1]
    image_transp = np.reshape(image, (28, 28))
    plt.imshow(image_transp, interpolation='none')
    plt.show()