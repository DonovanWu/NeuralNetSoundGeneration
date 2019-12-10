Neural Network Sound Generation
========

This project is inspired by this video: https://www.youtube.com/watch?v=FsVSZpoUdSU

Actually I first saw it on Bilibili: https://www.bilibili.com/video/av5497117

From the description of the video, it sounds like the author converted the raw audio bytes to base64 (or some invented encoding) and put it into an existing text-generation neural network. So I want to improve from the video in the following ways:
1. Convert the audio to spectrogram first, so that the neural network can directly learn frequency features, and also, since high-frequency band is unimportant, they can be thrown away and consequentially reduce feature space, speeding up the network
2. Write the actual neural network, have actual control of the hyperparameters, and potentially find a better model that suits the task
  * Note: the neural network shall still take raw audio data as input and be trained without other assistants, which is to say, it is not going to be a text-to-speech network, so the quality of the audio generated is probably going to be crappier than those networks, but I hope to achieve better results than what the video demonstrated

## Prerequisites

tensorflow==1.14.0

## Usages

Character-based LSTM:

```
python3 cblstm-like.py [-h] [--n_samples N_SAMPLES] [--sample_every N_EPOCHS]
                       sample

positional arguments:
  sample

optional arguments:
  -h, --help            show this help message and exit
  --n_samples N_SAMPLES
                        How many samples to generate each time model is
                        sampled.
  --sample_every N_EPOCHS
```

LSTM autoencoder:

```
python3 lstm-autoencoder.py [-h] [--n_samples N_SAMPLES]
                            [--sample_duration SECONDS] [--n_average N]
                            operation sample

positional arguments:
  operation             'train' or 'test'
  sample

optional arguments:
  -h, --help            show this help message and exit
  --n_samples N_SAMPLES
                        How many samples to generate when sampling from the
                        model.
  --sample_duration SECONDS
  --n_average N         [Experimental] For each sample, take average value of
                        N encodings. Default to 1, effectively disabling this
                        feature.
```

## Dev Notes

I was surpised by how powerful the open-source machine learning libraries are nowadays, as long as you have the right know-how. I was able to write a character-based text generation network and convert it to suit this project's question within a bit more than 100 lines of code.

## Current State

### Character-based LSTM

This model inputs a segment of frequency band and predicts the next frequency band.

Sadly I do not know how to properly "sample" from this network, because the output layer is not a distribution that represents categorical data.

However, the network can still learn from music audios fine (usually) and generate some sound that doesn't loop too quickly. It almost always just output silence for speech audios.

### LSTM Autoencoder

Again, this model takes segments of frequency band as input.

The autoencoder is kind of a failure as for now... Acutally it works as expected, as an autoencoder. The network outputs an "impression" of what it learned sounds like, but when I take the average of the encodings of two samples, it basically outputs the "superposition" of the two samples, so it's nothing different from what I can do by using an audio editing software. I tried to shrink the bottleneck, but the result is the same.

I also tried to use a different input from training time during prediction time, but the network then pretty much just outputs noise.
