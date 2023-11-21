import os
from tqdm import tqdm

import torch
import numpy as np
from torch.utils.data import Dataset

from src.utils.text import text_to_sequence


def get_data_to_buffer(data_path, mel_ground_truth, pitch_path, energy_path, alignment_path, text_cleaners, batch_expand_size, limit):
    buffer = []
    text = []
    with open(data_path, "r", encoding="utf-8") as f:
        text = []
        for line in f.readlines()[:limit]:
            text.append(line)

    for i in tqdm(range(len(text)), "get_data_to_buffer"):
        mel_gt_name = os.path.join(mel_ground_truth, "ljspeech-mel-%05d.npy" % (i + 1))
        mel_gt_target = np.load(mel_gt_name)
        duration = np.load(os.path.join(alignment_path, str(i) + ".npy"))
        character = text[i][0 : len(text[i]) - 1]
        character = np.array(text_to_sequence(character, text_cleaners))
        pitch_gt_name = os.path.join(pitch_path, "ljspeech-pitch-%05d.npy" % (i + 1))
        pitch_gt_target = np.load(pitch_gt_name).astype(np.float32)
        energy_gt_name = os.path.join(energy_path, "ljspeech-energy-%05d.npy" % (i + 1))
        energy_gt_target = np.load(energy_gt_name)

        character = torch.from_numpy(character)
        duration = torch.from_numpy(duration)
        mel_gt_target = torch.from_numpy(mel_gt_target)
        pitch_gt_target = torch.from_numpy(pitch_gt_target)
        energy_gt_target = torch.from_numpy(energy_gt_target)

        buffer.append(
            {
                "text": character,
                "duration": duration,
                "mel_target": mel_gt_target,
                "pitch": pitch_gt_target,
                "energy": energy_gt_target,
                "batch_expand_size": batch_expand_size,
            }
        )

    return buffer


class LJSpeechDataset(Dataset):
    def __init__(self, dir, text_cleaners, batch_expand_size, limit=None, **kwargs):
        self.buffer = get_data_to_buffer(
            data_path=(dir + "/train.txt"),
            mel_ground_truth=(dir + "/mels"),
            pitch_path=(dir + "/pitch"),
            energy_path=(dir + "/energy"),
            alignment_path=(dir + "/alignments"),
            text_cleaners=text_cleaners,
            batch_expand_size=batch_expand_size,
            limit=limit,
        )
        self.buffer = self.buffer[:limit]

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        return self.buffer[idx]
