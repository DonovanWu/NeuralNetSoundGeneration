import argparse
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import wavfile
from scipy import fftpack
from scipy.signal import butter, filtfilt

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout
from tensorflow.keras.callbacks import LambdaCallback


# This is not "sampling" the output because I can't, so it is not supposed to be done in this way.
# However, I can't figure out how to introduce "distribution" into the output of this model
def generate(n_frames=200, exclude_initial=True, diversity=1.0):
    i = np.random.randint(0, len(spectrogram) - n_per_group)
    result = spectrogram[i:i + n_per_group].copy()

    for i in range(n_frames):
        pred = model.predict(np.reshape(result[-n_per_group:], (1, n_per_group, n_feat)))
        y = pred[-1]
        result.append(y)

    if exclude_initial:
        result = result[n_per_group:]

    # inverse-transform spectrogram back to waveform
    n_pad = len(f_filter) - len(f[f_filter])
    for idx, val in enumerate(result):
        waveform = fftpack.idct(np.concatenate([val.flatten(), [0] * n_pad]), type=2, norm='ortho')
        waveform *= gain
        waveform = waveform.astype('int16')
        result[idx] = waveform

    result = np.array(result, dtype='int16')
    result = result.flatten()
    return result


def sample_callback(epoch, *args):
    if (epoch + 1) % sample_every != 0:
        return
    print('Sampling...')
    for i in range(n_samples):
        result = generate()
        filename = 'sample-ep%d-%d.wav' % (epoch + 1, i + 1)
        wavfile.write(filename, rate, result)
        print('Saved sampled sound to %s' % filename)
    filename = 'model-ep%d.h5' % (epoch + 1)
    model.save(filename)
    print('Model saved to %s' % filename)


def highpass_filter(ydata, wavepass, frequency):
    b, a = butter(2, wavepass * 2 / frequency, btype='high', analog=False, output='ba')
    return filtfilt(b, a, ydata)


parser = argparse.ArgumentParser()
parser.add_argument('sample')
parser.add_argument('--n_samples', action='store', type=int, default=3, required=False,
                    help='How many samples to generate each time model is sampled.')
parser.add_argument('--sample_every', action='store', type=int, default=25, required=False,
                    metavar='N_EPOCHS')
args = parser.parse_args()

sample = args.sample
n_samples = args.n_samples
sample_every = args.sample_every

# hyperparameters
gain = 2048         # to make loss value not that crazy (to prevent gradient explosion)
frame_width = 20    # this is in milliseconds
n_per_group = 2
step = 1

# load data and process parameters
rate, data = wavfile.read(sample)
if len(data.shape) > 1:    # multiple channels
    data = data.T[0]
frame_width = round(rate * frame_width / 1000)    # convert to number of points
data = data.astype('float32') / gain
data = highpass_filter(data, wavepass=20, frequency=rate)

# split data into frames and calculate spectrogram
spectrogram = []
end = frame_width
f = fftpack.fftfreq(frame_width, d=1 / rate)
f_filter = np.logical_and(f >= 0, f < 10000)
while end <= len(data):
    frame = data[end - frame_width:end]
    
    spectro = fftpack.dct(frame, type=2, norm='ortho')
    spectro = spectro[f_filter]
    spectrogram.append(spectro)
    
    end += frame_width

# turn spectrogram data into training and testing data
X, y = [], []
for i in range(0, len(spectrogram) - n_per_group, step):
    X.append(spectrogram[i:i + n_per_group])
    y.append(spectrogram[i + n_per_group])
X = np.array(X)
y = np.array(y)

# build model
n_feat = len(f[f_filter])
model = Sequential([
    LSTM(512, return_sequences=True, input_shape=(n_per_group, n_feat)),
    Dropout(0.1),
    LSTM(512),
    Dropout(0.1),
    Dense(n_feat)
])
model.compile(loss='mse', optimizer='adam')
model.summary()

print('Training LSTM model (stateless)...')
history = model.fit(X, y, batch_size=64, epochs=200, shuffle=True,
                    callbacks=[LambdaCallback(on_epoch_end=sample_callback)])
model.save('model.h5')
print('Saved to model.h5')
