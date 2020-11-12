import numpy as np
import time
import os
import json
import pickle as pkl
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt


class OnsetModel(tf.Module):

    def __init__(self, name=None, **kwargs):
        super().__init__(**kwargs)
        self.layers = [layers.Conv2D(10, (7, 3), activation='relu',
                                     input_shape=(15, 80, 3),
                                     data_format='channels_last'),
                       layers.MaxPool2D(pool_size=(1, 3), strides=3),
                       layers.Conv2D(20, (3, 3), activation='relu',
                                     data_format='channels_last'),
                       layers.MaxPool2D(pool_size=(1, 3), strides=3),
                       layers.Flatten(),
                       layers.Dense(256, activation='relu'),
                       layers.Dense(128, activation='relu'),
                       layers.Dense(1, activation='sigmoid')]

    def compile(self, loss, opt):
        self.loss = loss
        self.opt = opt

    def convert(self, song):
        ret = []
        for i in range(8, len(song) - 7):
            ret.append(song[i-8:i+7])
        ret = np.array(ret)
        return tf.Variable(ret)

    def __call__(self, inputs):
        inputs = self.convert(inputs)
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs

    def train(self, x, y, lr):
        y = tf.constant(y[8:-7])
        with tf.GradientTape() as t:
            current_loss = self.loss(y, model(x))
        grad = t.gradient(current_loss, self.trainable_variables)
        return grad, current_loss

    def fit_song(self, x, y, epochs, lr=1):
        for epoch in range(epochs):
            loss = self.train(x, y, lr)
            print(f"epoch: {epoch}\tloss: {loss}")

    def fit_dir(self, train_dir, epochs, lr=1):
        index = [line[:-1] for line in open(f"{train_dir}/index.txt")]
        total = len(index)
        for epoch in range(epochs):
            for i in range(len(index)):
                name = index[i]
                x = pkl.load(open(f"{train_dir}/{name}.pkl", "rb"))
                y = json.load(open(f"{train_dir}/{name}.bpm"))
                current_grad, current_loss = self.train(x, y, lr)
                if i == 0:
                    grad = current_grad
                else:
                    grad = (tf.math.add(var1, var2) for var1, var2 in zip(grad,
                                                                          current_grad))
                print(f"epoch: {epoch+1}\ttraining on: {name}\t({i+1}/{total})")
            grad = (tf.cast(dvar/len(index), dvar.dtype) for dvar in grad)
            self.opt.apply_gradients(zip(grad, self.trainable_variables))


if __name__ == "__main__":

    name = "A Happy Death"

    audio = pkl.load(open(f"./dataset_ddr/{name}.pkl", "rb"))
    bpm = json.load(open(f"./dataset_ddr/{name}.bpm", "r"))
    metadata = json.load(open(f"./dataset_ddr/{name}.metadata", "r"))

    model = OnsetModel()
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    opt = tf.keras.optimizers.Adam()
    model.compile(loss, opt)

    t0 = time.perf_counter()
    model.fit_dir("./dataset_ddr", 2, 1)
    t1 = time.perf_counter()
    print(f"time elapsed: {t1-t0}")

    pred = model(audio)
    print(sum(pred))
    print(pred.shape)
    pkl.dump(model.trainable_variables,
             open("weights.sav", "wb"))
    plt.plot(range(len(audio) - 15), pred)
    #plt.plot(range(len(audio) - 15),
             #bpm[8:-7], alpha=.2)
    plt.show()
