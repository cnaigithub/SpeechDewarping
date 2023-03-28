import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
import os
import argparse
import numpy as np
from tqdm import tqdm
import glob
import time as time
from scipy.io import wavfile

from hparams import get_hparams, add_hparams
from model import Tacotron2
from data_utils import TextMelLoader
import audio


device = None
MAX_WAV_VALUE = 32768.0

def load_model(hparams):
    model = Tacotron2(hparams).cuda()
    return model


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


def save_wav(wav, path, sr, max_wav_value=32767):
	wav *= max_wav_value / max(0.01, np.max(np.abs(wav)))
	wavfile.write(path, sr, wav.astype(np.int16))
        

def infer(checkpoint_path, text, dir_path, hparams, args):
    trainset = TextMelLoader(None, hparams)
    
    with open(text, 'r', encoding='utf-8') as f:
        lines = [x.strip().split('|') for x in f.readlines()]
    print('# data:', len(lines))

    checkpoint_dict = torch.load(checkpoint_path)
    model = load_model(hparams)
    model.load_state_dict(checkpoint_dict['state_dict'], strict=False)
    _ = model.cuda().eval()

    os.makedirs(dir_path, exist_ok=True)
    for i, line in enumerate(tqdm(lines)):
        print(line[1])
        gt_path = line[0]

        with torch.no_grad():
            inputs, gt_mel, wav_paths = trainset.get_mel_text_pair(line)
            inputs = inputs.unsqueeze(0).to(device)
            
            _, mel_outputs_postnet, _, alignments = model.inference(inputs)
            output_mel = mel_outputs_postnet.cpu().detach().numpy()
            mel = torch.FloatTensor(output_mel).to(device)
            if len(mel.shape) == 3:
                mel = mel.squeeze()
            x = mel.unsqueeze(0)
                
            x = x.to(device)
            audio_path = os.path.join(dir_path, f"synthesis_{i}.wav")
            os.makedirs(os.path.dirname(audio_path), exist_ok=True)
            
            mel = x.squeeze(0).cpu().numpy()
            total_wav = audio.inv_mel_spectrogram(mel, hparams)
            save_wav(total_wav, audio_path, hparams.sampling_rate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    hparams = add_hparams(parser)
    parser.add_argument('-c', '--checkpoint', type=str,help='checkpoint path', default=None)
    parser.add_argument('-o', '--out_filename', type=str, help='output filename', default='results')
    parser.add_argument('-t', '--text_path',type=str,help='txt file path', default='filelists/kss_final/test.txt')
    args = parser.parse_args()
    hparams = get_hparams(args,parser)

    hparams.sampling_rate = 16000
    hparams.filter_length = 2048
    hparams.hop_length = 200
    hparams.win_length = 800
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    infer(args.checkpoint, args.text_path, args.out_filename, hparams, args)
