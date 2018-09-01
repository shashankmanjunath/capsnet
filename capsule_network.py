from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


np.random.seed(2018)


class CapsuleNetwork:
    """Class that implements a capsule network on MNIST"""
    def __init__(self):
        self.X = tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32, name="X")


if __name__ == "__main__":
    mnist = input_data.read_data_sets("../capsnet/data/")

    n_samples = 5

    plt.figure(figsize=(n_samples * 2, 3))

    for index in range(n_samples):
        plt.subplot(1, n_samples, index+1)
        sample_image = mnist.train.images[index].reshape(28, 28)
        plt.imshow(sample_image, cmap="binary")
        plt.axis("off")

    plt.show()
