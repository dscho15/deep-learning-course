import torchvision.datasets as dataset
import os

import torch as torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as tf
import torch.nn.functional as functional

import net


if __name__ == '__main__':

    #
    dataset_train = dataset.MNIST(root = os.path.join(os.path.dirname(__file__), "dataset"), 
                                  train = True, 
                                  transform=tf.Compose([
                                                    tf.ToTensor()
                                                ])
                                  )

    #
    train_loader = torch.utils.data.DataLoader(dataset_train,
                                                batch_size=64,
                                                shuffle=True)
    
    #
    encoder = net.MConstrastEncoder(1)
    projector = net.MConstrastProjector(36, 64)

    # 
    augmentations = tf.Compose([
                                    tf.GaussianBlur((3, 3)),
                                    tf.RandomResizedCrop((28, 28))
                                ])

    for batch, (x, y) in enumerate(train_loader):

        x_1 = augmentations(x)
        x_2 = augmentations(x)

        y_1 = encoder(x_1)
        y_2 = encoder(x_2)

        z_1 = projector(y_1)
        z_2 = projector(y_2)

        z_1 = functional.normalize(z_1)
        z_2 = functional.normalize(z_2)

        s = z_1 @ z_2.T

        
        
    
    
