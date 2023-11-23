import argparse
import json
import os
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm

import src.model as module_model
from src.utils import DEFAULT_SR
from src.utils.parse_config import ConfigParser
from src.utils.text import text_to_sequence
from waveglow import get_wav, get_waveglow


def main(config, args):
    logger = config.get_logger("test")

    # define cpu or gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        logger.error("Error, supported cuda only!")

    # build model architecture
    model = config.init_obj(config["arch"], module_model)
    logger.info(model)
    logger.info("Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)
    logger.info("Checkpoint has been loaded.")
    model = model.to(device)
    model.eval()

    waveglow = get_waveglow(args.waveglow)

    results_dir = "test_model/results/"
    os.makedirs(results_dir, exist_ok=True)

    with open(args.input, "r") as f:
        texts = [text.strip() for text in f.readlines()]
    tokenized_texts = [text_to_sequence(t, ["english_cleaners"]) for t in texts]

    def generate_save_audio(i, text, tokenized_text, length_control, pitch_control, energy_control):
        src_seq = torch.tensor(tokenized_text, device=device).unsqueeze(0)
        src_pos = torch.arange(1, len(tokenized_text) + 1, device=device).unsqueeze(0)

        mel_prediction = model(
            src_seq=src_seq,
            src_pos=src_pos,
            length_control=length_control,
            pitch_control=pitch_control,
            energy_control=energy_control,
        )["mel_prediction"]
        wav = get_wav(mel_prediction.transpose(1, 2), waveglow, sampling_rate=DEFAULT_SR).unsqueeze(0)

        prefix_name = f"{i+1:04d}"
        suffix_name = f"l{round(length_control, 2)}-p{round(pitch_control, 2)}-e{round(energy_control, 2)}"
        with open(f"{results_dir}/{prefix_name}-text-{suffix_name}.txt", "w") as fout:
            fout.write(text)
        torchaudio.save(f"{results_dir}/{prefix_name}-audio-{suffix_name}.wav", wav, sample_rate=DEFAULT_SR)

    if args.test:
        for i, (text, tokenized_text) in tqdm(enumerate(zip(texts, tokenized_texts)), desc="inference"):
            generate_save_audio(i, text, tokenized_text, 1, 1, 1)
            for control in [0.8, 1.2]:
                generate_save_audio(i, text, tokenized_text, control, 1, 1)
                generate_save_audio(i, text, tokenized_text, 1, control, 1)
                generate_save_audio(i, text, tokenized_text, 1, 1, control)
                generate_save_audio(i, text, tokenized_text, control, control, control)

    else:
        for i, (text, tokenized_text) in tqdm(enumerate(zip(texts, tokenized_texts)), desc="inference"):
            generate_save_audio(i, text, tokenized_text, args.length_control, args.pitch_control, args.energy_control)

    logger.info("Audios have been generated.")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "-c",
        "--config",
        default="configs/test.json",
        type=str,
        help="Config path.",
    )
    args.add_argument(
        "-t",
        "--test",
        default=False,
        type=bool,
        help="Create multiple audio variants with different length, pitch and energy.",
    )
    args.add_argument(
        "-l",
        "--length-control",
        default=1,
        type=float,
        help="Increase or decrease audio speed.",
    )
    args.add_argument(
        "-p",
        "--pitch-control",
        default=1,
        type=float,
        help="Increase or decrease audio pitch.",
    )
    args.add_argument(
        "-e",
        "--energy-control",
        default=1,
        type=float,
        help="Increase or decrease audio energy.",
    )
    args.add_argument(
        "-cp",
        "--checkpoint",
        default="test_model/tts-checkpoint.pth",
        type=str,
        help="Checkpoint path.",
    )
    args.add_argument(
        "-i",
        "--input",
        default="test_model/input.txt",
        type=str,
        help="Input texts path.",
    )
    args.add_argument(
        "-w",
        "--waveglow",
        default="waveglow/pretrained_model/waveglow_256channels.pt",
        type=str,
        help="Waveglow weights path.",
    )
    args = args.parse_args()

    model_config = Path(args.config)
    with model_config.open() as fin:
        config = ConfigParser(json.load(fin))

    main(config, args)
