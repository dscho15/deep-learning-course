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
- ``57.92% accuracy`` on CIFAR test data.
- ``76.02% accuracy`` on CIFAR with small CNN and data-transformations.

# Lecture 4

Challenge
- Try to solve CIFAR10 with your own numpy (MLP) neural network, i.e. implement backprop, gradient descent, batch.

----------------------------------------------------------------

Results by using a single-layer, which is "handmade" in numpy:
- ``59.04% accuracy`` on CIFAR test data (not fine-tuned).

# Lecture 5

Challenge
- The exercise is clearly stated in the slides

----------------------------------------------------------------

- ``62.25% accuracy`` on CIFAR test data (not fine-tuned).

# Lecture 7

Challenge
- Create a CNN and try it on CIFAR10.

----------------------------------------------------------------

- ``89.47% accuracy`` on CIFAR with large CNN and data-transformations.

# Lecture 8

Challenge
- Create an autoencoder for celebA 
- Create a VAE for mnist

----------------------------------------------------------------
- I somewhat skipped the exercise, and went for VAE on celebA.
- It seems very difficult to find the correct set of hyperparameters, which is needed to train a VAE.

# Lecture 9

Challenge
- Train an RNN to classify spam emails
- Trian an RNN to sort arrays

----------------------------------------------------------------
- I managed to do the first exercise, first, by using chars, secondly, by using words. Neither, does better than randomly guess i.e. 85% correct.
