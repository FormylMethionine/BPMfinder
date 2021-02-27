import tensorflow as tf
from model import BeatCNN
import matplotlib.pyplot as plt
import time

# Create the model and adds an optimizer and a loss function
model = BeatCNN()
loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
opt = tf.keras.optimizers.Adam(learning_rate=1e-3)  # Best optimizer here ?
model.compile(loss, opt)

# Training params
epochs = 2000
batchSize = 16

# actual training of the model
t0 = time.perf_counter()
val_loss, train_loss = model.fit("./dataset_ddr/train",
                                 "./dataset_ddr/val",
                                 "./dataset_ddr/test",
                                 epochs,
                                 batch_size=batchSize)
t1 = time.perf_counter()
print(f"time elapsed: {t1-t0}")

# Saving the weights and plotting the loss
model.save(f"./weights/{epochs}_{batchSize}.wgh")
f4 = plt.figure(1)
plt.plot(range(epochs), val_loss)
plt.plot(range(epochs), train_loss)
plt.show()
