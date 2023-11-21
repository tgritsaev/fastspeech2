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

    if args.device is None:
        # define cpu or gpu if possible
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

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

    for i, (text, tokenized_text) in tqdm(enumerate(zip(texts, tokenized_texts)), desc="inference"):
        src_seq = torch.tensor(tokenized_text, device=device).unsqueeze(0)
        src_pos = torch.arange(1, len(tokenized_text) + 1, device=device).unsqueeze(0)

        mel_prediction = model(
            src_seq=src_seq,
            src_pos=src_pos,
            length_control=args.length_control,
            pitch_control=args.pitch_control,
            energy_control=args.energy_control,
        )["mel_prediction"]
        wav = get_wav(mel_prediction.transpose(1, 2), waveglow, sampling_rate=DEFAULT_SR).unsqueeze(0)

        prefix_name = f"{i+1:04d}_l{round(args.length_control, 2)}"
        suffix_name = f"p{round(args.pitch_control, 2)}_e{round(args.energy_control, 2)}"
        with open(f"{results_dir}/{prefix_name}-text-{suffix_name}.txt", "w") as fout:
            fout.write(text)
        torchaudio.save(f"{results_dir}/{prefix_name}-audio-{suffix_name}.wav", wav, sample_rate=DEFAULT_SR)

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
        default="test_model/checkpoint.pth",
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
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="Device, select 'cuda' or 'cpu'.",
    )
    args = args.parse_args()

    model_config = Path(args.config)
    with model_config.open() as fin:
        config = ConfigParser(json.load(fin))

    main(config, args)
