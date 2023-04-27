# Speech De-warping
PyTorch implementation of our [paper](https://arxiv.org/abs/2303.15669) "Unsupervised Pre-training for Data-Efficient Text-to-Speech on Low Resource Languages", ICASSP 2023.
Demo audio samples are available at the [demo page](https://cnaigithub.github.io/SpeechDewarping).

> **Abstract:** 
> Neural text-to-speech (TTS) models can synthesize natural human speech when trained on large amounts of transcribed speech.
> However, collecting such large-scale transcribed data is expensive.
> This paper proposes an unsupervised pre-training method for a sequence-to-sequence TTS model by leveraging large untranscribed speech data.
> With our pre-training, we can remarkably reduce the amount of paired transcribed data required to train the model for the target downstream TTS task. 
> The main idea is to pre-train the model to reconstruct de-warped mel-spectrograms from warped ones, which may allow the model to learn proper temporal assignment relation between input and output sequences.
> In addition, we propose a data augmentation method that further improves the data efficiency in fine-tuning.
> We empirically demonstrate the effectiveness of our proposed method in low-resource language scenarios, achieving outstanding performance compared to competing methods.
> The code and audio samples are available at: https://github.com/cnaigithub/SpeechDewarping

<!-- <strong> The repository is currently under construction.</strong> -->
The code is based on the [Tacotron 2](https://github.com/NVIDIA/tacotron2) repository.

## Installation
We tested our code in Ubuntu 20.04, CUDA 11.1 and Python 3.7.11 enviroment with A6000 GPUs.
```
conda create -n dewarp python=3.7.11
conda activate dewarp
pip install -r requirements.txt
pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```


## Dataset
For the unsupervised pre-training, we use speech data of 'train-clean-100' subset of the LibriTTS dataset.
To fine-tune the model with the transcribed speech, we use the KSS dataset for Korean and the LJspeech dataset for English.
The filelists of the datasets can be found in `./filelists`.

For custom datasets, follow the given filelist format for each line of the file.
- Pre-training: {Audio file path}|{Audio duration in seconds}
- Fine-training: {Audio file path}|{Text}


## Training
For each training scheme, refer to the explanation of the hyperparameter options in `./hparams.py` and set the options accordingly.
Example configuration files for each scheme are provided in `./filelists/example_hparams`.
```
# Unsupervised pre-training with speech data (Speech de-warping)
python train.py -o {Output folder to save checkpoints and logs}

# Fine-tuning with transcribed speech data
python train.py -o {Output folder to save checkpoints and logs} -c {Path of pre-trained checkpoint} --warm_start
```

## Inference
After fine-tuning, the checkpoint can be used for TTS inference.
```
python inference.py -c {Path to fine-tuned checkpoint} -o {output folder to save audio results} -t {filelist containing text to inference}
```

## Pre-trained Checkpoints
We provide the following checkpoints:
1. Pre-trained with Speech De-warping [(link)](https://drive.google.com/file/d/1lgSaJFKjHB7G9y1Rk1_3I-jW3c3PqS5H/view?usp=sharing)
2. Fine tuned from the above checkpoint, using SegAug with 0.5 shard of KSS data [(link)](https://drive.google.com/file/d/1t_np4ZpGmglrHWAuUS3lZc-zlbY8E_Pi/view?usp=sharing)

## Citation
```bibtex
@inproceedings{park2023icassp,
  title={Unsupervised Pre-training for Data-Efficient Text-to-Speech on Low Resource Languages},
  author={Park, Seongyeon and Song, Myungseo and Kim, Bohyung and Oh, Tae-Hyun},
  booktitle={ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2023},
  organization={IEEE}
}
```

