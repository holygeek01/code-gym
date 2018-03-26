from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import time

sample_rate, samples = wavfile.read('temp.wav')
frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
nfft = 256
fs = 256
pxx, freqs, bins, im = plt.specgram(samples, nfft,fs)
plt.axis('off')
plt.savefig('spec.png',
                    dpi=100, # Dots per inch
                    frameon='false',
                    aspect='normal',
                    bbox_inches='tight',
                    pad_inches=0) # Spectrogram saved as a .png

print("Training Data Generated...")
