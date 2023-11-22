# Text to speech with FastSpeech2 

[FastSpeech2 article](https://arxiv.org/pdf/2006.04558.pdf) and [FastSpeech article](https://arxiv.org/pdf/1905.09263.pdf).

## Example
https://github.com/tgritsaev/fastspeech2/assets/34184267/0598f598-337f-4947-a76c-a084c4e36453

## Installation guide

1. Use python3.9
```shell
conda create -n fastspeech2 python=3.9 && conda activate fastspeech2
```
2. Install libraries
```shell
pip3 install -r requirements.txt
```
3. Download data
```shell
bash scripts/download_data.sh
```
4. Preprocess data: save pitch and energy
```shell
python3 scripts/preprocess_data.py
```
5. Download my final FastSpeech2 checkpoint
```shell
python3 scripts/download_checkpoint.py
```

## Train 

1. Train
```shell
python3 train.py -c configs/train.json
```
Final model was trained with `train.json` config.

## Test

1. Test
```shell
python3 test.py
```
`test.py` include such arguments:
* Config path: `-c, --config, default="configs/test.json"`
* Increase or decrease audio speed: `-l, --length-control, default=1`
* Increase or decrease audio pitch: `-p, --pitch-control, default=1`
* Increase or decrease audio energy: `-e, --energy-control, default=1`
* Checkpoint path: `-cp, --checkpoint, default="test_model/tts-checkpoint.pth"`
* Input texts path: `-i, --input, test_model/input.txt`
* Waveglow weights path: `-w, --waveglow, default="waveglow/pretrained_model/waveglow_256channels.pt"`

## Wandb Report

https://api.wandb.ai/links/tgritsaev/rkir8sp9 (English only)

## Credits

This repository is based on a heavily modified fork
of [pytorch-template](https://github.com/victoresque/pytorch-template) repository. 
FastSpeech2 impementation is based on the code from HSE "Deep Learning in Audio" course [seminar](https://github.com/XuMuK1/dla2023/blob/2023/week07/seminar07.ipynb) and official [FastSpeech2 repository](https://github.com/ming024/FastSpeech2).
