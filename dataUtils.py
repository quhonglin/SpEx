import os
import librosa
import numpy as np


def sample_fixed_length_target(target_wav, min_duration, n_samples):
    if len(target_wav) < (min_duration * n_samples):
        distance = min_duration * n_samples - len(target_wav)
        return np.append(target_wav, np.zeros(distance))
    else:
        return target_wav

def sample_fixed_length_interfer(target_wav, interfer_wav):
    if len(target_wav) < len(interfer_wav):
        start = np.random.randint(len(interfer_wav) - len(target_wav) + 1)
        end = start + len(target_wav)
        return interfer_wav[start:end]
    elif len(target_wav) > len(interfer_wav):
        distance = len(target_wav) - len(interfer_wav)
        return np.append(interfer_wav, np.zeros(distance))
    else:
        return interfer_wav

def mix(target_wav, interfer_wav):
    """
    混合语音
    :param target_wavs: 目标语音
    :param interfer_wavs: 干扰语音
    :return: 混合语音
    """
    mix_wav = target_wav + interfer_wav
    mix_wav = 2 * (mix_wav - np.min(mix_wav)) / (np.max(mix_wav) - np.min(mix_wav)) - 1
    mix_wav = quieter(mix_wav)
    return mix_wav


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
