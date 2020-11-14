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

pkl.dump(model.save(),
         open("weights.sav", "wb"))

name = "A Happy Death"

audio = pkl.load(open(f"./dataset_ddr/train/{name}.pkl", "rb"))
bpm = json.load(open(f"./dataset_ddr/train/{name}.bpm", "r"))
metadata = json.load(open(f"./dataset_ddr/train/{name}.metadata", "r"))

name2 = "Anti the Holic"

audio2 = pkl.load(open(f"./dataset_ddr/test/{name2}.pkl", "rb"))
bpm2 = json.load(open(f"./dataset_ddr/test/{name2}.bpm", "r"))
metadata2 = json.load(open(f"./dataset_ddr/test/{name2}.metadata", "r"))

name3 = "Hold Release Rakshasa and Carcasses"

audio3 = pkl.load(open(f"./dataset_ddr/train/{name3}.pkl", "rb"))
bpm3 = json.load(open(f"./dataset_ddr/train/{name3}.bpm", "r"))
metadata3 = json.load(open(f"./dataset_ddr/train/{name3}.metadata", "r"))

pred = model(audio)
pred2 = model(audio2)
pred3 = model(audio3)

f1 = plt.figure(1)
plt.plot(range(len(audio) - 15), pred)
plt.plot(range(len(audio) - 15),
         bpm[8:-7], alpha=.2)
f2 = plt.figure(2)
plt.plot(range(len(audio2) - 15), pred2)
plt.plot(range(len(audio2) - 15),
         bpm2[8:-7], alpha=.2)
f3 = plt.figure(3)
plt.plot(range(len(audio3) - 15), pred3)
plt.plot(range(len(audio3) - 15),
         bpm3[8:-7], alpha=.2)
f4 = plt.figure(4)
plt.plot(range(epochs), val_loss)
f5 = plt.figure(5)
plt.plot(range(epochs), train_loss)
plt.show()
