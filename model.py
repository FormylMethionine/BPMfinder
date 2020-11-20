import numpy as np
import time
import json
import pickle as pkl
import tensorflow as tf
from tensorflow.keras import layers
from math import floor
import matplotlib.pyplot as plt


class OnsetModel(tf.Module):

    def __init__(self, name=None, **kwargs):
        super().__init__(**kwargs)
        self.layers = [layers.Conv2D(10, (7, 3), activation='relu',
                                     input_shape=(15, 80, 3),
                                     data_format='channels_last'),
                       layers.MaxPool2D(pool_size=(1, 3), strides=3),
                       layers.Conv2D(20, (3, 3), activation='relu'),
                       layers.MaxPool2D(pool_size=(1, 3), strides=3),
                       layers.Flatten(),
                       layers.Dense(256, activation='relu'),
                       layers.Dropout(.5),
                       layers.Dense(128, activation='relu'),
                       layers.Dropout(.5),
                       layers.Dense(1, activation='sigmoid'),
                       layers.Dropout(.5)]

    def compile(self, loss, opt):
        self.loss = loss
        self.opt = opt

    def convert(self, song):
        ret = []
        for i in range(7, len(song) - 8):
            ret.append(song[i-7:i+8])
        ret = np.array(ret)
        return tf.Variable(ret)

    def __call__(self, inputs):
        inputs = self.convert(inputs)
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs

    def train(self, x, y):
        y = tf.constant(y[7:-8])
        with tf.GradientTape() as t:
            current_loss = self.loss(y, self.__call__(x))
        grad = t.gradient(current_loss, self.trainable_variables)
        #self.opt.apply_gradients(zip(grad, self.trainable_variables))
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
            train_loss = 0
            np.random.shuffle(index_train)
            for i, name in enumerate(index_train[:batch_size]):
                # training
                x = pkl.load(open(f"{train_dir}/{name}.pkl", "rb"))
                y = json.load(open(f"{train_dir}/{name}.bpm", "r"))
                curr_grad, curr_train_loss = self.train(x, y)
                train_loss += curr_train_loss
                if i == 0:
                    grad = curr_grad
                else:
                    grad += curr_grad
                print(f"epoch: {epoch+1}/{epochs}\t" +
                      f"training on: {name}\t({i+1}/{batch_size})")
            grad = (tf.math.divide(dvar, batch_size) for dvar in grad)
            self.opt.apply_gradients(zip(grad, self.trainable_variables))
            train_loss /= batch_size
            train_loss_ret.append(train_loss)

            val_loss = 0
            for i in range(total_val):  # validation
                name = index_val[i]
                x = pkl.load(open(f"{val_dir}/{name}.pkl", "rb"))
                y = json.load(open(f"{val_dir}/{name}.bpm", "r"))
                val_loss += self.evaluate(x, y)
                print(f"epoch: {epoch+1}/{epochs}\t" +
                      f"evaluating: {name}\t({i+1}/{total_val})")
            val_loss_ret.append(val_loss/total_val)

        test_loss = 0
        for i in range(total_test):  # test
            name = index_test[i]
            x = pkl.load(open(f"{test_dir}/{name}.pkl", "rb"))
            y = json.load(open(f"{test_dir}/{name}.bpm", "r"))
            test_loss += self.evaluate(x, y)
            print(f"testing: {name}\t({i+1}/{total_test})")

        test_loss /= total_test
        print(f"test loss:\t{test_loss}")
        return val_loss_ret, train_loss_ret

    def get_weights(self):
        ret = []
        for layer in self.layers:
            ret.append(layer.get_weights())
        return ret

    def set_weights(self, W):
        for layer, w in zip(self.layers, W):
            layer.set_weights(w)

    def save(self, path):
        W = self.get_weights()
        pkl.dump(W, open(path, "wb"))

    def load(self, path):
        W = pkl.load(open(path, "rb"))
        self.set_weights(W)


if __name__ == "__main__":

    model = OnsetModel()
    model.compile(tf.keras.losses.BinaryCrossentropy(),
                  tf.keras.optimizers.Adam())
    model(pkl.load(open("dataset_ddr/train/A Happy Death.pkl", "rb")))
    #model.fit("dataset_ddr/train", "dataset_ddr/val", "dataset_ddr/test",
              #1)
    model.load(pkl.load(open("weights.sav", "rb")))
    pkl.dump(model.save(),
             open("weights.sav", "wb"))
