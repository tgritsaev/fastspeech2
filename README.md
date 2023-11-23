# Text to speech with FastSpeech2 

[FastSpeech2 article](https://arxiv.org/pdf/2006.04558.pdf) and [FastSpeech article](https://arxiv.org/pdf/1905.09263.pdf).

## Example
Inference result is audio, but Github supports only video+audio formats.

https://github.com/tgritsaev/fastspeech2/assets/34184267/80b357d5-6a8f-492d-a550-d8c83645e2f2

You can also download a folder with [tts-results](https://drive.google.com/drive/folders/1U0REGoBFfei30iCxaMrHEaIx0zVPMSIO?usp=share_link) from Google Drive, it includes 27 audios with different length, pitch and energy for the first three inputs from `test_model/input.txt`.

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
1. Run for training 
```shell
python3 train.py -c configs/train.json
```
Final model was trained with `train.json` config.

## Test
1. Run for testing
```shell
python3 test.py
```
`test.py` include such arguments:
* Config path: `-c, --config, default="configs/test.json"`
* Create multiple audio variants with different length, pitch and energy `-t, --test, default=False`
* Increase or decrease audio speed: `-l, --length-control, default=1`
* Increase or decrease audio pitch: `-p, --pitch-control, default=1`
* Increase or decrease audio energy: `-e, --energy-control, default=1`
* Checkpoint path: `-cp, --checkpoint, default="test_model/tts-checkpoint.pth"`
* Input texts path: `-i, --input, test_model/input.txt`
* Waveglow weights path: `-w, --waveglow, default="waveglow/pretrained_model/waveglow_256channels.pt"`

Results will be saved in the `test_model/results`, you can see example in this folder.

## Wandb Report

[https://api.wandb.ai/links/tgritsaev/rkir8sp9](https://wandb.ai/tgritsaev/dla3_text_to_speech/reports/Text-to-speech-with-FastSpeech2--Vmlldzo2MDU2MjU5?accessToken=4bu09pvt6ik0wpse85z5cnckki7q0pcdfo8aug40gw942v1h1jcf1gtp2vfo8w58) (English only)

## Credits

This repository is based on a heavily modified fork
of [pytorch-template](https://github.com/victoresque/pytorch-template) repository. 
FastSpeech2 impementation is based on the code from HSE "Deep Learning in Audio" course [seminar](https://github.com/XuMuK1/dla2023/blob/2023/week07/seminar07.ipynb) and official [FastSpeech2 repository](https://github.com/ming024/FastSpeech2).
