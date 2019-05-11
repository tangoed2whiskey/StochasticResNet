import numpy as np 
import tensorflow as tf
from dense_net import dense_net
from resnet import resnet

def main(dataset=None,method=None,**kwargs):
    if dataset=='mnist' or dataset=='MNIST':
        mnist = tf.keras.datasets.mnist
        (x_train, y_train),(x_test, y_test) = mnist.load_data()
        num_classes=10
        if method=='resnet':
            x_train = x_train.reshape(x_train.shape+(1,))
            x_test = x_test.reshape(x_test.shape+(1,))
    elif dataset=='cifar' or dataset=='CIFAR':
        cifar = tf.keras.datasets.cifar10
        (x_train, y_train),(x_test, y_test) = cifar.load_data()
        # idx = np.random.choice(np.arange(len(x_train)), 5000, replace=False)
        # x_train, y_train = x_train[idx], y_train[idx]
        num_classes=10
    else:
        print('Unknown dataset {}, please give a known dataset'.format(dataset))
        exit()

    if method=='dense':
        these_kwargs = {key:kwargs[key] for key in ['stochastic','dropout','epochs'] if key in kwargs}
        dense_net(x_train=x_train,x_test=x_test,y_train=y_train,y_test=y_test,num_classes=num_classes,**these_kwargs)
    elif method=='resnet':
        these_kwargs = {key:kwargs[key] for key in ['epochs','batch_size','subtract_pixel_mean','n'] if key in kwargs}
        resnet(x_train=x_train,x_test=x_test,y_train=y_train,y_test=y_test,num_classes=num_classes,**these_kwargs)
    else:
        print('Unknown method {}, please give a known method'.format(method))
        exit()




if __name__=='__main__':
    main(dataset='mnist',method='dense',stochastic=False,dropout=False,epochs=100)