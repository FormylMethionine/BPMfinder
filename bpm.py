from model import BeatCNN
from audio import analyze
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle as pkl
import json
from scipy import signal
import numpy as np
import sys

# ***
# Getting the predictions
# ***

model = BeatCNN()

path = sys.argv[1]
audio = analyze(path)
model.load("./weights/2000_16.wgh")

pred = model(audio).numpy().astype('float64')[:, 0]
f, Pxx = signal.periodogram(pred)
f = 5172*f

# ***
# Finding the bpm
# ***

pks = signal.find_peaks(Pxx)[0]  # finding peaks
pks = [f[pks] for pks in sorted(pks, key=lambda x: Pxx[x], reverse=True)
       if 50 <= f[pks] <= 500][:10]
bpm = pks[0]
i = 1
for i in range(1, 5):  # Converting the highest peak to a bpm
    new_bpm = bpm/i
    if int(new_bpm) <= 200 and round(new_bpm) in list(map(round, pks)):
        break
print(f"Predicted BPM: {round(new_bpm)}")

# ***
# Plots
# ***

f1, ax1 = plt.subplots()
ax1.plot(f, Pxx)
ax1.set_xlim(50, 500)
ax1.set_ylim(0, max(Pxx))
ax1.set_title(f"Periodogram of {path.split('/')[-1]}")
ax1.set_xlabel("BPMs")

f2, ax2 = plt.subplots()
spec = ax2.imshow(audio[3000:6000, :, 1].T, cmap="Greys", aspect='auto')
ax2.set_title(f"Mel spectrogram and predicted beats of {path.split('/')[-1]}")
ax2.set_ylim(79, 0)
ax2.set_xlabel('Time [frame]')
ax2.set_ylabel('Frequency bands')
cb = f2.colorbar(spec, orientation='horizontal', shrink=.7)

secax = ax2.secondary_xaxis('top',
                            functions=(lambda x: 11.6e-3*x,
                                       lambda x: 11.6e-3*x))
secax.set_xlabel('Time [s]')

ax3 = ax2.twinx()
ax3.plot(pred[3000:6000], color='red', alpha=1)
ax3.set_ylabel('Predicted probability of presence of a beat',
               color='red')
ax3.tick_params('y', labelcolor='red')

#ax4 = ax2.twinx()
#ax4.plot(annot[2000:4000], color='orange', alpha=.5)
#ax4.axis('off')

plt.show()
