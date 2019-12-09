import argparse
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import wavfile
from scipy import fftpack
from scipy.signal import butter, filtfilt

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping


def highpass_filter(ydata, wavepass, frequency):
    b, a = butter(2, wavepass * 2 / frequency, btype='high', analog=False, output='ba')
    return filtfilt(b, a, ydata)


def wavfile2spectrogram(filename, channel=0):
    # load data and process parameters
    rate, data = wavfile.read(filename)
    assert rate == 44100       # temporary solution lol
    if len(data.shape) > 1:    # multiple channels
        data = data.T[channel]
    data = data.astype('float32') / gain
    data = highpass_filter(data, wavepass=20, frequency=rate)

    frame_width = round(rate * frame_width_ms / 1000)

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
    n_pad = len(f_filter) - len(f[f_filter])
    n_feat = len(f[f_filter])

    return spectrogram, rate, n_pad, n_feat


parser = argparse.ArgumentParser()
parser.add_argument('operation', help="'train' or 'test'")
parser.add_argument('sample')
parser.add_argument('--n_samples', action='store', type=int, default=3, required=False,
                    help="How many samples to generate when sampling from the model.")
parser.add_argument('--sample_duration', action='store', type=float, default=3, required=False,
                    metavar='SECONDS')
parser.add_argument('--n_average', action='store', type=int, default=1, required=False, metavar='N',
                    help="[Experimental] For each sample, take average value of N encodings. " \
                         "Default to 1, effectively disabling this feature.")
args = parser.parse_args()

operation = args.operation.lower()
sample = args.sample
n_samples = args.n_samples
sample_duration = args.sample_duration
n_average = args.n_average

if operation not in ('train', 'test'):
    raise ValueError('Unknown operation: %s' % operation)

# hyperparameters
gain = 2048         # to make loss value not that crazy (to prevent gradient explosion)
frame_width_ms = 20
latent_layers = {'lstm1': 256, 'lstm2': 64}
batch_size = 64
epochs = 100

# convert to number of points
sample_lenth = round(sample_duration * 1000 / frame_width_ms)

spectrogram, rate, n_pad, n_feat = wavfile2spectrogram(sample, channel=0)

# turn spectrogram data into training and testing data
X = []
X = np.array(spectrogram)
X = X.reshape(len(X), 1, n_feat)

# build model
if operation == 'train':
    model = Sequential([
        LSTM(latent_layers['lstm1'], activation='elu', return_sequences=True, input_shape=(1, n_feat), name='lstm_enc1'),
        LSTM(latent_layers['lstm2'], activation='elu', name='lstm_enc2'),
        RepeatVector(1, name='encoding'),
        LSTM(latent_layers['lstm2'], activation='elu', return_sequences=True, name='lstm_dec1'),
        LSTM(latent_layers['lstm1'], activation='elu', return_sequences=True, name='lstm_dec2'),
        TimeDistributed(Dense(n_feat), name='output')
    ])
    model.compile(loss='mse', optimizer='adam')
    model.summary()

    model.fit(X, X, batch_size=batch_size, epochs=epochs, shuffle=True,
              callbacks=[EarlyStopping(monitor='loss', min_delta=0.01, patience=1)])
    model.save('model.h5')
    print('Saved to model.h5')

# prediction testing
encoder = Sequential([
    LSTM(latent_layers['lstm1'], activation='elu', return_sequences=True, input_shape=(1, n_feat), name='lstm_enc1'),
    LSTM(latent_layers['lstm2'], activation='elu', name='lstm_enc2'),
    RepeatVector(1, name='encoding')
])
decoder = Sequential([
    LSTM(latent_layers['lstm2'], activation='elu', return_sequences=True, name='lstm_dec1', input_shape=(1, latent_layers['lstm2'])),
    LSTM(latent_layers['lstm1'], activation='elu', return_sequences=True, name='lstm_dec2'),
    TimeDistributed(Dense(n_feat), name='output')
])
encoder.load_weights('model.h5', by_name=True)
decoder.load_weights('model.h5', by_name=True)

for i in range(n_samples):
    encodings = []
    for j in range(n_average):
        start_idx = np.random.randint(0, len(spectrogram) - sample_lenth)
        X_test = np.array(spectrogram[start_idx:start_idx + sample_lenth])
        X_test = X_test.reshape(sample_lenth, 1, n_feat)
        y_pred = encoder.predict(X_test)
        encodings.append(y_pred)
    encodings = np.mean(encodings, axis=0)
    y_pred = decoder.predict(encodings)

    result = []
    for val in y_pred:
        waveform = fftpack.idct(np.concatenate([val.flatten(), [0] * n_pad]), type=2, norm='ortho')
        waveform *= gain
        waveform = waveform.astype('int16')
        result.append(waveform)
    result = np.array(result, dtype='int16')
    result = result.flatten()
    wavfile.write('sample-output-%d.wav' % (i + 1), rate, result)
    print('Saved sampled file to sample-output-%d.wav' % (i + 1))
