import neuralnetwork as nn
import numpy as np
import matplotlib.pyplot as plt 

# .npz is a foramt for storing multiple numpy arrays.

with np.load('mnist.npz') as data:
    training_images = data['training_images']
    training_labels = data['training_labels']


# display the images....
#plt.imshow(training_images[0].reshape(28,28), cmap = 'gray')
#plt.show()
#print(training_labels[0])




layer_sizes = (784,15,10) 
x = np.ones((layer_sizes[0],1))


net = nn.NeuralNetwork(layer_sizes)
net.print_accuracy(training_images,training_labels)


print( int(net.cost_function(training_images, training_labels)) ) 
