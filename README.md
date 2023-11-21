# Text to speech with FastSpeech2 https://arxiv.org/pdf/2006.04558.pdf

## Installation guide

1. Install libraries
```shell
pip install -r requirements.txt
```
2. Download Librispeech and create Mixture dataset
```shell
python create_dataset.py -c create_dataset.json
```
3. If you want to test my solution quality, download my speech separation checkpoint `ss-checkpoint.pth` from the https://drive.google.com/drive/folders/14dn7NIHOfOoIUm_hCkZ7RUHhniErGvzp?usp=sharing. Optional, if you want to measure WER and CER, download my audio speech recognition checkpoint, named `asr-checkpoint.pth`, from the same link.

## Train 

1. General training pipeline
```shell
python train.py -c src/configs/config.json
```
2. Reproduce my final train setup
```shell
python train.py -c path_to_config
```

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

## Docker

You can use this project with docker. Quick start:

```bash 
docker build -t my_src_image . 
docker run \
   --gpus '"device=0"' \
   -it --rm \
   -v /path/to/local/storage/dir:/repos/asr_project_template/data/datasets \
   -e WANDB_API_KEY=<your_wandb_api_key> \
	my_src_image python -m unittest 
```

Notes:

* `-v /out/of/container/path:/inside/container/path` -- bind mount a path, so you wouldn't have to download datasets at
  the start of every docker run.
* `-e WANDB_API_KEY=<your_wandb_api_key>` -- set envvar for wandb (if you want to use it). You can find your API key
  here: https://wandb.ai/authorize
