import os
import subprocess
import json

import librosa
import librosa.display
import numpy as np

from utils import chars

DIRNAME = os.path.dirname(__file__)
SONG_NAME = 'hard_times'
PATH_TO_AUDIO = os.path.join(DIRNAME, '../audio/hard_times.mp3')
PATH_TO_FRAMES = os.path.join(DIRNAME, '../frames', SONG_NAME + '/')
PATH_TO_RENDERS = os.path.join(DIRNAME, '../renders', SONG_NAME + '/')
PATH_TO_DATA_OUT = os.path.join(DIRNAME, '../data/', SONG_NAME + '/')

DURATION = 30.0
SAMPLE_RATE = 44100
FPS = 30
POWER = 11
FFT_SIZE = pow(2, POWER)
HOP_LENGTH = round(SAMPLE_RATE / FPS)  # todo: better way to make this an int?
ZERO_PADDING = '%06d'

"""
D - a 2D array of frames x freq-amplitude buckets [[A1, A2, ...], [A1, A2, ...], ...]
amount - the proportion of the n - 1 frame to mix with the nth frame
"""


def applySmoothing(D, amount=.85):
    frame_count = len(D)
    bin_count = len(D[0])
    for i in range(1, frame_count):
        for j in range(1, bin_count):
            if i == 1:
                prev = 0
            else:
                prev = D[i - 1][j]
            D[i][j] = D[i][j] * (1 - amount) + prev * amount


def exportFrame(i, f, y):
    filename = (ZERO_PADDING % i) + '.png'
    print(f'Exporting frame {filename}')
    plt.clf()
    plt.plot(f, y)
    plt.tight_layout()
    plt.xscale('log')
    plt.savefig(os.path.join(PATH_TO_FRAMES, filename),
                bbox_inches='tight')


def processFrames(data, frequencies, parallelism=None):
    if parallelism:
        p = Pool(parallelism)
        p.starmap(exportFrame, [(index, frequencies, frame)
                                for index, frame in enumerate(data)])
    else:
        for index, frame in enumerate(data):
            exportFrame(index, frame, frequencies)


def writeOut(data):
    for k, v in data.items():
        with open(os.path.join(PATH_TO_DATA_OUT, k + '.json'), 'w') as out_file:
            out_file.write(json.dumps(v.tolist()))


def processAudio(f_method='fft', b_method='times'):

    # Get raw PCM data
    y, sr = librosa.load(
        PATH_TO_AUDIO,
        duration=DURATION,
        sr=SAMPLE_RATE,
        mono=True
    )

    # Separate harmonics and percussives into two waveforms
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    # Beat track on the percussive signal
    tempo, beat_frames = librosa.beat.beat_track(y=y_percussive,
                                                 sr=SAMPLE_RATE)

    if b_method == 'times':
        B = librosa.frames_to_time(beat_frames, sr=SAMPLE_RATE)
    else:
        B = beat_frames

    if f_method == 'fft':
        D = librosa.stft(
            y,
            n_fft=FFT_SIZE,
            hop_length=HOP_LENGTH,
            center=True
        )
        F = librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=FFT_SIZE)

    elif f_method == 'cqt':
        D = librosa.cqt(
            y,
            sr=SAMPLE_RATE,
            hop_length=HOP_LENGTH,
            n_bins=84,
            bins_per_octave=12,
            fmin=28
        )
        F = librosa.cqt_frequencies(84, 28, 12)

    D = np.transpose(librosa.amplitude_to_db(np.abs(D), ref=np.max))

    return {
        "d": D,
        "f": F,
        "b": B,
    }


def checkPaths(path_array):
    for path in path_array:
        if not os.path.exists(path):
            os.makedirs(path)


def main():

    checkPaths([PATH_TO_AUDIO, PATH_TO_DATA_OUT,
                PATH_TO_FRAMES, PATH_TO_RENDERS])
    audio_objects = processAudio(f_method='fft', b_method='times')
    applySmoothing(audio_objects["d"], .85)
    writeOut(audio_objects)

    # processFrames(audio_objects["d"], audio_objects["f"])
    # renderVideo('test_cqt', 630, 474, FPS * DURATION, FPS)


if __name__ == '__main__':
    main()
