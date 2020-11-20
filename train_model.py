import tensorflow as tf
from model import BeatCNN
import matplotlib.pyplot as plt
import time

model = BeatCNN()
loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(loss, opt)

epochs = 600
batchSize = 16
t0 = time.perf_counter()
val_loss, train_loss = model.fit("./dataset_ddr/train",
                                 "./dataset_ddr/val",
                                 "./dataset_ddr/test",
                                 epochs,
                                 batch_size=batchSize)
t1 = time.perf_counter()
print(f"time elapsed: {t1-t0}")

model.save(f"./weights/{epochs}_{batchSize}.wgh")
f4 = plt.figure(1)
plt.plot(range(epochs), val_loss)
plt.plot(range(epochs), train_loss)
plt.show()
