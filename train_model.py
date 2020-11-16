import pickle as pkl
import json
import tensorflow as tf
from model import OnsetModel
import matplotlib.pyplot as plt
import time

model = OnsetModel()
loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
opt = tf.keras.optimizers.Adam()
model.compile(loss, opt)

epochs = 50
t0 = time.perf_counter()
val_loss, train_loss = model.fit("./dataset_ddr/train",
                                 "./dataset_ddr/val",
                                 "./dataset_ddr/test",
                                 epochs)
t1 = time.perf_counter()
print(f"time elapsed: {t1-t0}")

model.save("weights.sav")
f4 = plt.figure(1)
plt.plot(range(epochs), val_loss)
plt.plot(range(epochs), train_loss)
plt.show()
