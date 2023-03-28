import random
import os
import numpy as np
import torch
import torch.utils.data

from g2p_en import G2p
from g2pk import G2p as G2p_korean

import layers
from utils import load_wav_to_torch, load_filepaths_and_text
from text import _symbols_to_sequence
from text_kr import text_to_sequence as text_to_sequence_korean

import audio

g2p = G2p()
g2p_korean = G2p_korean()

speaker2idx_libri100 = {6147: 0, 7178: 1, 1034: 2, 1040: 3, 1553: 4, 19: 5, 5652: 6, 7190: 7, 3607: 8, 26: 9, 27: 10, 32: 11, 4640: 12, 8226: 13, 6181: 14, 39: 15, 40: 16, 4137: 17, 3112: 18, 8747: 19, 2092: 20, 5163: 21, 5678: 22, 8238: 23, 1069: 24, 1578: 25, 5688: 26, 1081: 27, 7226: 28, 1594: 29, 60: 30, 1088: 31, 6209: 32, 8770: 33, 4160: 34, 5703: 35, 4680: 36, 5192: 37, 1098: 38, 587: 39, 78: 40, 3664: 41, 83: 42, 87: 43, 1624: 44, 2136: 45, 89: 46, 1116: 47, 8797: 48, 3168: 49, 7264: 50, 4195: 51, 7780: 52, 103: 53, 7278: 54, 2159: 55, 625: 56, 7794: 57, 3699: 58, 4214: 59, 118: 60, 7800: 61, 8312: 62, 5750: 63, 125: 64, 6272: 65, 2691: 66, 8324: 67, 8838: 68, 7302: 69, 2182: 70, 3723: 71, 3214: 72, 7312: 73, 5778: 74, 2196: 75, 150: 76, 669: 77, 5789: 78, 1183: 79, 6818: 80, 3235: 81, 163: 82, 3240: 83, 3242: 84, 4267: 85, 5808: 86, 7859: 87, 4788: 88, 6836: 89, 696: 90, 3259: 91, 6848: 92, 196: 93, 198: 94, 7367: 95, 200: 96, 4297: 97, 5322: 98, 1737: 99, 201: 100, 4813: 101, 2764: 102, 1743: 103, 211: 104, 1235: 105, 730: 106, 5339: 107, 1246: 108, 6367: 109, 6880: 110, 3807: 111, 226: 112, 8419: 113, 4830: 114, 229: 115, 8425: 116, 7402: 117, 5867: 118, 233: 119, 1263: 120, 6385: 121, 2289: 122, 4340: 123, 4853: 124, 3830: 125, 248: 126, 250: 127, 4859: 128, 254: 129, 2817: 130, 4362: 131, 6925: 132, 5390: 133, 8975: 134, 6415: 135, 5393: 136, 8465: 137, 3857: 138, 8468: 139, 2836: 140, 7447: 141, 2843: 142, 289: 143, 4898: 144, 6437: 145, 3879: 146, 298: 147, 4397: 148, 302: 149, 3374: 150, 1841: 151, 307: 152, 6454: 153, 1334: 154, 4406: 155, 311: 156, 831: 157, 322: 158, 839: 159, 1867: 160, 6476: 161, 2893: 162, 1355: 163, 8014: 164, 2384: 165, 5456: 166, 7505: 167, 1363: 168, 332: 169, 7511: 170, 2391: 171, 4441: 172, 5463: 173, 7517: 174, 2910: 175, 2911: 176, 1898: 177, 3947: 178, 3436: 179, 6000: 180, 2416: 181, 3440: 182, 8051: 183, 374: 184, 887: 185, 8063: 186, 6529: 187, 4481: 188, 6531: 189, 2436: 190, 8580: 191, 6019: 192, 1926: 193, 2952: 194, 5514: 195, 909: 196, 3982: 197, 911: 198, 3983: 199, 403: 200, 7059: 201, 405: 202, 8088: 203, 7067: 204, 412: 205, 5022: 206, 8095: 207, 3486: 208, 8609: 209, 8098: 210, 6563: 211, 7078: 212, 1447: 213, 426: 214, 1963: 215, 8108: 216, 2989: 217, 4014: 218, 1455: 219, 6064: 220, 4018: 221, 1970: 222, 8629: 223, 8630: 224, 5561: 225, 5049: 226, 8123: 227, 446: 228, 6078: 229, 6081: 230, 3526: 231, 1992: 232, 7113: 233, 458: 234, 460: 235, 2514: 236, 4051: 237, 7635: 238, 2002: 239, 2518: 240, 2007: 241, 1502: 242, 481: 243, 7148: 244, 5104: 245, 4088: 246}

def bshall_melspectrogram(wav, hparams):
    D = audio._stft(wav, hparams)
    S = audio._amp_to_db(audio._linear_to_mel(np.abs(D)**hparams.magnitude_power, hparams), hparams) - hparams.ref_level_db

    return audio._normalize(S, hparams)

def speech_warping(mel_spec, duration_list, hparams):
    resize_list = []
    for start, end in duration_list:
        orig_fragment = mel_spec[:, start:end].unsqueeze(0).unsqueeze(0)
        resized_fragment = torch.nn.functional.interpolate(orig_fragment, size=(hparams.n_mel_channels, hparams.mel_resize_step_num), mode='bilinear')
        resize_list.append(resized_fragment)
    warped_mel = torch.cat(resize_list, dim = -1).squeeze(0).squeeze(0)
    return warped_mel
  
def segaug(mel_spec, hparams):
    T = mel_spec.shape[1]
    num_of_segments = round(T * hparams.naive_resize_factor)
    if num_of_segments == 1: duration_list = [(0, T)]
    else:
        seperators = [0] + random.sample(list(range(1, T)), num_of_segments-1) + [T]
        seperators.sort()
        duration_list = [(seperators[i], seperators[i+1]) for i in range(0, num_of_segments)]

    resize_list = []    
    for start, end in duration_list:
        orig_fragment = mel_spec[:, start:end].unsqueeze(0).unsqueeze(0)
        randomstep = (random.random()*4 + 1)/3
        resized_fragment = torch.nn.functional.interpolate(orig_fragment, size=(hparams.n_mel_channels, max(1, int(randomstep*(end-start)))), mode='bilinear')
        resize_list.append(resized_fragment)
    warped_mel = torch.cat(resize_list, dim = -1).squeeze(0).squeeze(0)
    return warped_mel

class TextMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, audiopaths_and_text, hparams):
        self.hp = hparams
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text) if audiopaths_and_text is not None else []
        self.sampling_rate = hparams.sampling_rate
        self.stft = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)
        random.seed(hparams.seed)
        random.shuffle(self.audiopaths_and_text)

    def duration_txt_to_duration_list(self, duration_txt_path):
        return_list = []
        with open(duration_txt_path, 'r') as infile:
            lines = [x.strip() for x in infile.readlines()]
            lines = [0] + [float(x) for x in lines]
            for idx in range(len(lines)-1):
                start_time, end_time = lines[idx], lines[idx+1]
                return_list.append((int(start_time * self.hp.sampling_rate / self.hp.hop_length), int(end_time * self.hp.sampling_rate / self.hp.hop_length)))
        return return_list

    def get_mel_text_pair(self, audiopath_and_text):
        if self.hp.text_finetune:
            audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
            if self.hp.iskorean:
                text = g2p_korean(text)
                text = torch.IntTensor(text_to_sequence_korean(text, ['korean_cleaners']))
            else:
                phone_list = g2p(text)
                text = torch.IntTensor(_symbols_to_sequence(phone_list, clean_symbols=True))
                
            if self.hp.mel_resize_step_num > 1:
                text = text.repeat_interleave(self.hp.mel_resize_step_num)
                
            mel, _ = self.get_mel(audiopath, None)

            if self.hp.segaug:
                mel = segaug(mel, self.hp)
                
            return text, mel, audiopath
        
        elif self.hp.speech_dewarping:
            audiopath, duration = audiopath_and_text[0], float(audiopath_and_text[-1])
            T = int(duration*self.hp.sampling_rate / self.hp.hop_length)
            num_of_segments = round(self.hp.naive_resize_factor*T)

            if num_of_segments <= 1: duration_list = [(0, T)]
            else:
                seperators = [0] + random.sample(list(range(1, T)), num_of_segments-1) + [T]
                seperators.sort()
                duration_list = [(seperators[i], seperators[i+1]) for i in range(0, num_of_segments)]
                
            mel, warped_mel = self.get_mel(audiopath, duration_list)
            if self.hp.concat_speaker_embedding and self.hp.num_speaker>1:
                return warped_mel, mel, torch.IntTensor([speaker2idx_libri100[int(audiopath.split('/')[-3])]]), audiopath
            return warped_mel, mel, audiopath
        
        elif self.hp.naive_speech_autoencoder:
            audiopath = audiopath_and_text[0]
            mel, _ = self.get_mel(audiopath, None)
            orig_fragment = mel.unsqueeze(0).unsqueeze(0)
            resized_mel = torch.nn.functional.interpolate(orig_fragment, size=(mel.shape[0], int(mel.shape[1]*self.hp.naive_resize_factor)), mode='bilinear').squeeze(0).squeeze(0)
            if self.hp.concat_speaker_embedding and self.hp.num_speaker>1:
                return resized_mel, mel, torch.IntTensor([speaker2idx_libri100[int(audiopath.split('/')[-3])]]), audiopath
            return resized_mel, mel, audiopath
            
    def get_mel(self, filename, duration_list):            
        audio, sampling_rate = load_wav_to_torch(filename, self.hp)
        if sampling_rate != self.stft.sampling_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(
                sampling_rate, self.stft.sampling_rate))

        melspec = bshall_melspectrogram(audio, self.hp)
        melspec = torch.FloatTensor(melspec.astype(np.float32))
        melspec = torch.squeeze(melspec, 0)

        if duration_list == None or self.hp.text_finetune:
            return melspec, None
        warped_mel = speech_warping(melspec, duration_list, self.hp)
        return melspec, warped_mel
    
    def __getitem__(self, index):
        return self.get_mel_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)

class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, n_frames_per_step, hparams):
        self.hp = hparams
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text, mel-spectrogram, and wav_path
        PARAMS
        ------
        batch: [text_normalized, mel_normalized, wav_path]
        """
        # Right zero-pad all one-hot text sequences to max input length
        if self.hp.text_finetune:
            input_lengths, ids_sorted_decreasing = torch.sort(
                torch.LongTensor([len(x[0]) for x in batch]),
                dim=0, descending=True)
        elif self.hp.speech_dewarping or self.hp.naive_speech_autoencoder:
            input_lengths, ids_sorted_decreasing = torch.sort(
                torch.LongTensor([x[0].shape[1] for x in batch]),
                dim=0, descending=True)            
            
        # print(input_lengths)
        max_input_len = input_lengths[0]
        num_mels = batch[0][1].size(0)

        if self.hp.text_finetune:
            text_padded = torch.LongTensor(len(batch), max_input_len)
            text_padded.zero_()
        elif self.hp.speech_dewarping or self.hp.naive_speech_autoencoder:
            warped_mel_padded = torch.FloatTensor(len(batch), num_mels, max_input_len)
            warped_mel_padded.zero_()
            
        for i in range(len(ids_sorted_decreasing)):
            if self.hp.text_finetune:
                text = batch[ids_sorted_decreasing[i]][0]
                text_padded[i, :text.size(0)] = text
            elif self.hp.speech_dewarping or self.hp.naive_speech_autoencoder:
                warped_mel = batch[ids_sorted_decreasing[i]][0]
                warped_mel_padded[i, :, :warped_mel.size(1)] = warped_mel
            
        # Right zero-pad mel-spec
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()

        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        speaker_idxs = torch.IntTensor(len(batch))
        speaker_idxs.zero_()
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            output_lengths[i] = mel.size(1)
            if self.hp.concat_speaker_embedding and self.hp.num_speaker>1:
                speaker_idxs[i] = batch[ids_sorted_decreasing[i]][2]

        # Assume that wav_path is in last position.
        wav_paths = [batch[i][-1] for i in ids_sorted_decreasing]

        if self.hp.text_finetune:
            return text_padded, None, input_lengths, mel_padded, gate_padded, output_lengths, speaker_idxs, wav_paths
        elif self.hp.speech_dewarping or self.hp.naive_speech_autoencoder:
            return None, warped_mel_padded, input_lengths, mel_padded, gate_padded, output_lengths, speaker_idxs, wav_paths
            
    
