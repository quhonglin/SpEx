import os
import random
import librosa
import numpy as np
from tqdm import tqdm


def mix(target_wavs, interfer_wavs):
    """
    混合语音
    :param target_wavs: 目标语音
    :param interfer_wavs: 干扰语音
    :return: 混合语音
    """
    mix_wavs = [[] for i in range(len(target_wavs))]

    for i in tqdm(range(len(target_wavs)), desc="Mixing speakers"):
        for j in range(len(target_wavs[i])):
            if len(target_wavs[i][j]) < len(interfer_wavs[i][j]):
                short_len = len(target_wavs[i][j])
                long_len = len(interfer_wavs[i][j])
                while short_len < long_len:
                    target_wavs[i][j] = np.append(target_wavs[i][j], 0)
                    short_len += 1
            elif len(target_wavs[i][j]) > len(interfer_wavs[i][j]):
                short_len = len(interfer_wavs[i][j])
                long_len = len(target_wavs[i][j])
                while short_len < long_len:
                    interfer_wavs[i][j] = np.append(interfer_wavs[i][j], 0)
                    short_len += 1
            else:
                pass

            mix_wav = target_wavs[i][j] + interfer_wavs[i][j]
            mix_wav = 2 * (mix_wav - np.min(mix_wav)) / (np.max(mix_wav) - np.min(mix_wav)) - 1
            mix_wav = quieter(mix_wav)

            assert len(target_wavs[i][j]) == len(mix_wav)

            mix_wavs[i].append(mix_wav)

    return target_wavs, mix_wavs


def shuffle(wavs):
    """洗牌"""
    random.shuffle(wavs)
    for wav in wavs:
        random.shuffle(wav)
    return wavs


def quieter(wav):
    max_amp = np.max(np.abs(wav))
    return 0.38 * wav / max_amp


def data_save(data, sr, mix_save_path, refer_save_path, target_save_path, interfer1_save_path, interfer2_save_path=None):
    for i in range(len(data)):
        num = str(i + 1)
        librosa.output.write_wav(os.path.abspath(os.path.expanduser(mix_save_path)) + "/mix_" + num + ".wav", data[i][0], sr=sr)
        librosa.output.write_wav(os.path.abspath(os.path.expanduser(refer_save_path)) + "/refer_" + num + ".wav", data[i][1], sr=sr)
        librosa.output.write_wav(os.path.abspath(os.path.expanduser(target_save_path)) + "/target_" + num + ".wav", data[i][2], sr=sr)
        if interfer2_save_path:
            librosa.output.write_wav(os.path.abspath(os.path.expanduser(interfer1_save_path)) + "/interfer1_" + num + ".wav", data[i][3], sr=sr)
            librosa.output.write_wav(os.path.abspath(os.path.expanduser(interfer2_save_path)) + "/interfer2_" + num + ".wav", data[i][4], sr=sr)
            continue
        librosa.output.write_wav(os.path.abspath(os.path.expanduser(interfer1_save_path)) + "/interfer_" + num + ".wav", data[i][3], sr=sr)
