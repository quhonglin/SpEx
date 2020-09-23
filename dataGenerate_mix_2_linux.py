import glob
import copy
import math
import os
import random
import librosa
from tqdm import tqdm
from random import randrange
from dataUtils import mix, shuffle, data_save

if __name__ == "__main__":

    sr = 16000
    speaker_path = glob.glob(os.path.abspath(os.path.expanduser("~/dataset/TIMIT/*/*/*")))
    wavs_path = [glob.glob(path + "/*.wav") for path in speaker_path]
    wavs = []
    for speaker in tqdm(wavs_path, desc="Loading wavs"):
        wavs.append([librosa.load(wav, sr=sr)[0] for wav in speaker])

    # 目标语音 + 干扰语音 = 混合语音
    target_wavs = copy.deepcopy(wavs)
    interfer_wavs = copy.deepcopy(wavs)
    interfer_wavs = shuffle(interfer_wavs)

    # 参考语音
    refer_wavs = [[] for i in range(len(target_wavs))]
    for i in range(len(target_wavs)):
        for j in range(len(target_wavs[i])):
            rand = randrange(10)
            while rand == j:
                rand = randrange(10)
            refer_wavs[i].append(target_wavs[i][rand])

    target_2_wavs, mix_2_wavs = mix(target_wavs, interfer_wavs)

    assert len(mix_2_wavs) == len(refer_wavs) == len(target_2_wavs) == len(interfer_wavs)

    # 说话人混合的数据集
    mix_2 = []
    for i in range(len(mix_2_wavs)):
        for j in range(len(mix_2_wavs[i])):
            mix_2.append([mix_2_wavs[i][j], refer_wavs[i][j],
                               target_2_wavs[i][j], interfer_wavs[i][j]])

    assert len(mix_2)  == (len(wavs) * len(wavs[0]))

    # 分割训练集、验证集和测试集
    random.shuffle(mix_2)
    train_end = math.floor(len(mix_2)*0.6)
    dev_end = train_end + math.floor(len(mix_2)*0.2)
    train_2 = mix_2[: train_end]
    dev_2 = mix_2[train_end: dev_end]
    test_2 = mix_2[dev_end:]

    # 保存
    data_save(data=train_2, sr=sr,
              mix_save_path="~/dataset/mix_2/train/mix",
              refer_save_path="~/dataset/mix_2/train/refer",
              target_save_path="~/dataset/mix_2/train/target",
              interfer1_save_path="~/dataset/mix_2/train/interfer")

    data_save(data=dev_2, sr=sr,
              mix_save_path="~/dataset/mix_2/dev/mix",
              refer_save_path="~/dataset/mix_2/dev/refer",
              target_save_path="~/dataset/mix_2/dev/target",
              interfer1_save_path="~/dataset/mix_2/dev/interfer")

    data_save(data=test_2, sr=sr,
              mix_save_path="~/dataset/mix_2/test/mix",
              refer_save_path="~/dataset/mix_2/test/refer",
              target_save_path="~/dataset/mix_2/test/target",
              interfer1_save_path="~/dataset/mix_2/test/interfer")
