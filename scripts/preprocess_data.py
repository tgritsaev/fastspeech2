from tqdm import tqdm
from pathlib import Path
import logging

import numpy as np
from scipy import interpolate
import torch
import torchaudio
import pyworld as pw


logger = logging.getLogger("preprocess")

DATA_PATH = Path(__file__).absolute().resolve().parent.parent / "data"
WAV_PATH = DATA_PATH / "LJSpeech-1.1" / "wavs"
MEL_PATH = DATA_PATH / "mels"
PITCH_PATH = DATA_PATH / "pitch"
ENERGY_PATH = DATA_PATH / "energy"


def save_pitch():
    PITCH_PATH.mkdir(exist_ok=True, parents=True)

    wavs = []
    for file in WAV_PATH.iterdir():
        wavs.append(file.name)
    wavs.sort()

    min_pitch = np.inf
    max_pitch = -np.inf
    for i, wav_name in tqdm(enumerate(wavs), total=len(wavs)):
        mel = np.load(MEL_PATH / ("ljspeech-mel-%05d.npy" % (i + 1)))

        audio, sr = torchaudio.load(str(WAV_PATH / wav_name))
        audio = audio.to(torch.float64).numpy().sum(axis=0)

        frame_period = (audio.shape[0] / sr * 1000) / mel.shape[0]
        pitch, t = pw.dio(audio, sr, frame_period=frame_period)
        pitch = pw.stonemask(audio, pitch, t, sr)[: mel.shape[0]]
        nonzeros_ids = np.nonzero(pitch)
        x = np.arange(pitch.shape[0])[nonzeros_ids]
        values = (pitch[nonzeros_ids][0], pitch[nonzeros_ids][-1])
        f = interpolate.interp1d(x, pitch[nonzeros_ids], bounds_error=False, fill_value=values)
        pitch = f(np.arange(pitch.shape[0]))

        np.save(PITCH_PATH / ("ljspeech-pitch-%05d.npy" % (i + 1)), pitch)

        min_pitch = min(min_pitch, pitch.min())
        max_pitch = max(max_pitch, pitch.max())

    logger.info(f"min_pitch: {min_pitch}\nmax_pitch: {max_pitch}")


def save_energy():
    ENERGY_PATH.mkdir(exist_ok=True, parents=True)
    min_energy = np.inf
    max_energy = -np.inf
    for mel_path in tqdm(MEL_PATH.iterdir()):
        mel = np.load(mel_path)
        energy = np.linalg.norm(mel, axis=-1)
        np.save(ENERGY_PATH / mel_path.name.replace("mel", "energy"), energy)
        min_energy = min(min_energy, energy.min())
        max_energy = max(max_energy, energy.max())
    logger.info(f"min_energy: {min_energy}\nmax_energy: {max_energy}")


def main():
    save_pitch()
    save_energy()


if __name__ == "__main__":
    main()
