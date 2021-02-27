import numpy as np
import json
import pickle as pkl
import tensorflow as tf
from tensorflow.keras import layers


class BeatCNN(tf.Module):
    # Model for predicting if a frame is a beat or not

    def __init__(self, name=None, **kwargs):
        # Defining the model
        super().__init__(**kwargs)
        self.layers = [layers.Conv2D(10, (7, 3), activation='relu',
                                     input_shape=(15, 80, 3),
                                     data_format='channels_last',
                                     kernel_initializer='glorot_uniform',
                                     bias_initializer='zeros'),
                       layers.MaxPool2D(pool_size=(1, 3), strides=3),
                       layers.Conv2D(20, (3, 3), activation='relu',
                                     kernel_initializer='glorot_uniform',
                                     bias_initializer='zeros'),
                       layers.MaxPool2D(pool_size=(1, 3), strides=3),
                       layers.Flatten(),
                       layers.Dense(256, activation='relu',
                                    kernel_initializer='glorot_uniform',
                                    bias_initializer='zeros'),
                       layers.Dropout(.5),
                       layers.Dense(128, activation='relu',
                                    kernel_initializer='glorot_uniform',
                                    bias_initializer='zeros'),
                       layers.Dropout(.5),
                       layers.Dense(1, activation='sigmoid',
                                    kernel_initializer='glorot_uniform',
                                    bias_initializer='zeros'),
                       layers.Dropout(.5)]

    def compile(self, loss, opt):
        # Adding a loss function and an optimizer to the model
        self.loss = loss
        self.opt = opt

    def convert(self, song):
        # transforms a song to a (3D) tensor
        # that can be used by the model
        ret = []
        for i in range(7, len(song) - 8):
            ret.append(song[i-7:i+8])
        ret = np.array(ret, dtype=np.float32)
        return tf.Variable(ret)

    def __call__(self, inputs):
        inputs = self.convert(inputs)
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs

    def train(self, x, y):
        # returns gradient and loss for a (converted) song
        y = tf.constant(y[7:-8])
        with tf.GradientTape() as t:
            current_loss = self.loss(y, self.__call__(x))
        grad = t.gradient(current_loss, self.trainable_variables)
        return grad, current_loss

    def evaluate(self, x, y):
        y = tf.constant(y[7:-8])
        current_loss = self.loss(y, self.__call__(x))
        return current_loss

    def fit(self, train_dir, val_dir, test_dir, epochs, batch_size=32):
        index_train = [line[:-1] for line in open(f"{train_dir}/index.txt")]
        index_val = [line[:-1] for line in open(f"{val_dir}/index.txt")]
        index_test = [line[:-1] for line in open(f"{test_dir}/index.txt")]

        if batch_size == "full":
            batch_size = len(index_train)
        total_val = len(index_val)
        total_test = len(index_test)

        train_loss_ret = []
        val_loss_ret = []

        for epoch in range(epochs):

            # Training on a batch
            train_loss = 0
            np.random.shuffle(index_train)
            for i, name in enumerate(index_train[:batch_size]):
                # getting gradient for an element of the batch
                x = pkl.load(open(f"{train_dir}/{name}.pkl", "rb"))
                y = json.load(open(f"{train_dir}/{name}.bpm", "r"))
                curr_grad, curr_train_loss = self.train(x, y)
                train_loss += curr_train_loss
                if i == 0:
                    grad = curr_grad
                else:
                    grad += curr_grad  # Averaging gradient on the batch
                print(f"epoch: {epoch+1}/{epochs}\t" +
                      f"training on: {name}\t({i+1}/{batch_size})")
            grad = (tf.math.divide(dvar, batch_size) for dvar in grad)
            self.opt.apply_gradients(zip(grad, self.trainable_variables))
            train_loss /= batch_size
            train_loss_ret.append(train_loss)

            # Validation
            val_loss = 0
            for i in range(total_val):
                name = index_val[i]
                x = pkl.load(open(f"{val_dir}/{name}.pkl", "rb"))
                y = json.load(open(f"{val_dir}/{name}.bpm", "r"))
                val_loss += self.evaluate(x, y)
                print(f"epoch: {epoch+1}/{epochs}\t" +
                      f"evaluating: {name}\t({i+1}/{total_val})")
            val_loss_ret.append(val_loss/total_val)

        # Test
        test_loss = 0
        for i in range(total_test):
            name = index_test[i]
            x = pkl.load(open(f"{test_dir}/{name}.pkl", "rb"))
            y = json.load(open(f"{test_dir}/{name}.bpm", "r"))
            test_loss += self.evaluate(x, y)
            print(f"testing: {name}\t({i+1}/{total_test})")

        test_loss /= total_test
        print(f"test loss:\t{test_loss}")
        return val_loss_ret, train_loss_ret

    def get_weights(self):
        # Extracting sets from the model
        ret = []
        for layer in self.layers:
            ret.append(layer.get_weights())
        return ret

    def set_weights(self, W):
        # Setting weights from a list of weights
        # Calls the model on random data to initialize weights
        # A bit awkward, but not too slow
        rand = np.random.rand(100, 80, 3)
        self.__call__(rand)
        for layer, w in zip(self.layers, W):
            layer.set_weights(w)

    def save(self, path):
        # Saving weights in a file
        W = self.get_weights()
        pkl.dump(W, open(path, "wb"))

    def load(self, path):
        # Loading weights from a file
        W = pkl.load(open(path, "rb"))
        self.set_weights(W)
