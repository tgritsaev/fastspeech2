# Text to speech with FastSpeech2 

Based on https://arxiv.org/pdf/2006.04558.pdf

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
5. If you want to test my solution quality, download my speech separation checkpoint `ss-checkpoint.pth` from the https://drive.google.com/drive/folders/14dn7NIHOfOoIUm_hCkZ7RUHhniErGvzp?usp=sharing. Optional, if you want to measure WER and CER, download my audio speech recognition checkpoint, named `asr-checkpoint.pth`, from the same link.

## Train 

1. Train
```shell
python3 train.py -c configs/train.json
```
Final model was trained with `train.json` config.

## Test

1. Make sure that you created dataset and downloaded all needed checkpoints, the path to the dataset in config is correct.
2. If your config and checkpoint are placed as `test_model/config.json` and `test_model/checkpoint.pth` respectively, you can simply run
```shell
python test.py
```
3. Or you can specify paths and run 
```shell
python test.py -c path_to_config --ss_checkpoint path_to_ss_checkpoint
```
4. If you have your test dataset in the following format:
```shell
.
├── mix
│   ├── ID-mixed.wav
│   └── ...
├── refs
│   ├── ID-ref.wav
│   └── ...
└── targets
    ├── ID-target.wav
    └── ... 
```
Run
```shell
python test.py --test_data_folder path_to_data_folder
```
This approach does not support texts.

5. If you want to measure speech recognition model quality on my speech separation solution, use `test_model/asr_config.json` and run 
```shell
python test.py -c test_model/config_for_asr.json --ss_checkpoint path_to_ss_checkpoint --asr_checkpoint path_to_asr_checkpoint`.
```
6. If you want to test quality on segmented audio (100ms default window length), use `test_model/segmentation_config.json` and run 
```shell
python test.py -c test_model/segmentation_config.json -s window_len_in_seconds
```

## Wandb Report

https://api.wandb.ai/links/tgritsaev/rkir8sp9 (Russian only)

## Credits

This repository is based on a heavily modified fork
of [pytorch-template](https://github.com/victoresque/pytorch-template) repository. 
FastSpeech2 impementation is based on the code from HSE "Deep Learning in Audio" course [seminar](https://github.com/XuMuK1/dla2023/blob/2023/week07/seminar07.ipynb) and official [FastSpeech2 repository](https://github.com/ming024/FastSpeech2).
