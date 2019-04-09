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

layer_sizes = (3,5,10) 
x = np.ones((layer_sizes[0],1))

net = nn.NeuralNetwork(layer_sizes)
#net.print_accuracy(training_images,training_labels)

n1 = net.predict(x)
net.enable_constant()
n2 = net.predict(x)

print(n1)
print(n2)
