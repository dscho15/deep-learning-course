# Lecture 1

Challenge
- Try different values for k (passed to the KNeighborsClassifier constructor)- What can you achieve for MNIST and CIFAR-10?
- Visualize some of the errors (image, ground truth, predicted)

----------------------------------------------------------------

Bonus: 
- Try different metrics (passed to the KNeighborsClassifier constructor)
- Try extracting features by decomposing the images with PCA (sklearn

----------------------------------------------------------------

Results by using KNN and PCA, and data reduction by a factor of 1/6:
- MNIST results : ``0.9517%``
- CIFAR10 results : ``0.2889%``

# Lecture 2

Challenge
- Logistic regression in 2D, ``reaching 93% accuracy``.

----------------------------------------------------------------

Bonus:
- Logistic regression in 3D, ``reaching 91.6% accuracy``.
- Logistic regression on MNIST, ``reaching 91.4% accuracy with PCA=48``.

# Lecture 3

Challenge
- Get familiar with PyTorch
- Create a DL and try it on CIFAR10.

----------------------------------------------------------------

Results by using a 256-four-hiddenlayers net, heavy-dropout, large capacity:
- 57.92% accuracy on CIFAR test data.
- 76.02% accuracy on CIFAR with small CNN and data-transformations.

- ``Note to myself, remember to push the correct CNN net.``

# Lecture 4

Challenge
- Try to solve CIFAR10 with your own numpy (MLP) neural network, i.e. implement backprop, gradient descent, batch.

----------------------------------------------------------------

