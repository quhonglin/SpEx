import glob
import copy
import math
import os
import random
import librosa
from tqdm import tqdm
from random import randrange
from dataUtils import mix, data_save, sample_fixed_length_target, sample_fixed_length_interfer

if __name__ == "__main__":

    sr = 16000
    n_samples = 16000
    min_duration = 2
    speaker_path = glob.glob(os.path.abspath(os.path.expanduser("~/dataset/TIMIT/*/*/*")))
    wavs_path = [glob.glob(path + "/*.wav") for path in speaker_path]
    wavs = {}
    speaker_num = "speaker"

    for i, speaker in enumerate(tqdm(wavs_path, desc="Loading wavs")):
        speaker_num = "speaker" + str(i+1)
        wavs[speaker_num] = [librosa.load(wav, sr=sr)[0] for wav in speaker]

    # 目标语音 + 干扰语音 = 混合语音
    target_wavs = copy.deepcopy(wavs)
    interfer_wavs = copy.deepcopy(wavs)

    # 参考语音
    refer_wavs = {}
    for target_speaker in target_wavs.keys():
        refer_wavs[target_speaker] = []
        for i in range(len(target_wavs[target_speaker])):
            refer_wav_num = randrange(len(target_wavs[target_speaker]))
            while refer_wav_num == i:
                refer_wav_num = randrange(len(target_wavs[target_speaker]))
            refer_wavs[target_speaker].append(target_wavs[target_speaker][refer_wav_num])

    # 混合语音
    mix_wavs = {}
    for target_speaker in tqdm(target_wavs.keys(), desc="Mixing speakers"):

        interfer_speaker = random.sample(target_wavs.keys(), 1)[0]
        while target_speaker == interfer_speaker:
            interfer_speaker = random.sample(target_wavs.keys(), 1)[0]

        mix_wavs[target_speaker] = []

        for target_wav_num in range(len(target_wavs[target_speaker])):

            interfer_wav_num = random.randrange(len(interfer_wavs[interfer_speaker]))
            while interfer_wav_num == target_wav_num:
                interfer_wav_num = random.randrange(len(interfer_wavs[interfer_speaker]))

            target_wav = target_wavs[target_speaker][target_wav_num]
            interfer_wav = interfer_wavs[interfer_speaker][interfer_wav_num]

            target_wav = sample_fixed_length_target(target_wav, min_duration, n_samples)
            interfer_wav = sample_fixed_length_interfer(target_wav, interfer_wav)

            target_wavs[target_speaker][target_wav_num] = target_wav
            interfer_wavs[interfer_speaker][interfer_wav_num] = interfer_wav

            mix_wav = mix(target_wav, interfer_wav)
            mix_wavs[target_speaker].append(mix_wav)

            assert len(target_wav) == len(mix_wav)


    # 说话人混合的数据集
    mix_2_data = []
    for key in mix_wavs.keys():
        for i in range(len(mix_wavs[key])):
            assert len(mix_wavs[key][i]) == len(target_wavs[key][i])
            mix_2_data.append([mix_wavs[key][i], refer_wavs[key][i],
                               target_wavs[key][i], interfer_wavs[key][i]])

    # 分割训练集、验证集和测试集
    random.shuffle(mix_2_data)
    train_end = math.floor(len(mix_2_data) * 0.6)
    dev_end = train_end + math.floor(len(mix_2_data) * 0.2)
    train_2 = mix_2_data[: train_end]
    dev_2 = mix_2_data[train_end: dev_end]
    test_2 = mix_2_data[dev_end:]

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
