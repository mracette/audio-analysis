"""
ffmpeg command:
ffmpeg -r 30 -f image2 -s 630x474 -i ./frames/%03d.png -i ./audio/hard_times.mp3 -vcodec libx264 -pix_fmt yuv420p -frames 150 test.mp4

"""

import os
import subprocess

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

from utils import chars

DIRNAME = os.path.dirname(__file__)
PATH_TO_AUDIO = os.path.join(DIRNAME, '../audio/hard_times.mp3')
PATH_TO_FRAMES = os.path.join(DIRNAME, '../frames')
PATH_TO_RENDERS = os.path.join(DIRNAME, '../renders')

DURATION = 5.0
SR_MONO = 44100
FPS = 30
POWER = 11
FFT_SIZE = pow(2, POWER)
HOP_LENGTH = round(SR_MONO / FPS)  # todo: better way to make this an int?
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


def renderVideo(output_name, width, height, n_frames, fps):
    args = ['ffmpeg']
    args.append('-y')
    args.extend(['-framerate', str(fps)])
    args.extend(['-s', f'{width}x{height}'])
    args.extend(['-i', os.path.join(PATH_TO_FRAMES, ZERO_PADDING + '.png')])
    args.extend(['-i', PATH_TO_AUDIO])
    args.extend(['-vcodec', 'libx264'])
    args.extend(['-pix_fmt', 'yuv420p'])
    # if n_frames:
    #     args.extend(['-frames:v', str(n_frames)])
    args.append('-shortest')
    args.append(os.path.join(PATH_TO_RENDERS, output_name + '.mp4'))
    print(chars.info + 'Rendering video...')
    subprocess.Popen(args)
    print(chars.success + f'Video rendered to {PATH_TO_RENDERS}')


def exportFrames(D, frequencies):
    for i, f in enumerate(D):
        filename = (ZERO_PADDING % i) + '.png'
        print(f'Exporting frame {filename}')
        plt.clf()
        plt.plot(frequencies, f)
        plt.tight_layout()
        plt.xscale('log')
        plt.savefig(os.path.join(PATH_TO_FRAMES, filename),
                    bbox_inches='tight')


def processFft():
    y, sr = librosa.load(
        PATH_TO_AUDIO,
        duration=DURATION,
        sr=SR_MONO,
        mono=True
    )
    print(y)
    fft = librosa.stft(
        y,
        n_fft=FFT_SIZE,
        hop_length=HOP_LENGTH,
        center=True
    )
    frequencies = librosa.fft_frequencies(sr=SR_MONO, n_fft=FFT_SIZE)
    matrix = np.transpose(librosa.amplitude_to_db(np.abs(fft), ref=np.max))
    return {
        "d": matrix,
        "f": frequencies
    }


def main():
    # fft = processFft()
    # applySmoothing(fft["d"], .85)
    # exportFrames(fft["d"], fft["f"])
    renderVideo('test_smoothing', 630, 474, FPS * DURATION, FPS)


if __name__ == '__main__':
    main()
