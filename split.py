# misc script for moving files to train, test and val datasets
import os
import shutil
import numpy as np


def move(li, d):
    for name in li:
        shutil.move(f"{name}.bpm",
                    f"{d}/{name}.bpm")
        shutil.move(f"{name}.pkl",
                    f"{d}/{name}.pkl")
        shutil.move(f"{name}.metadata",
                    f"{d}/{name}.metadata")


index = [line[:-1] for line in open("index.txt", "r")]
np.random.shuffle(index)
train = index[:int(0.8*len(index))]
val = index[int(0.8*len(index)):int(0.9*len(index))]
test = index[int(0.9*len(index)):len(index)]

move(train, "train")
move(test, "test")
move(val, "val")
