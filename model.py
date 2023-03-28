import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
import random
import numpy as np
from math import sqrt

from utils import to_gpu, get_mask_from_lengths
from zoneoutrnn import ZoneoutRNN
from layers import ConvNorm, LinearNorm, LinearBN


class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size,
                 attention_dim):
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(2, attention_n_filters,
                                      kernel_size=attention_kernel_size,
                                      padding=padding, bias=False, stride=1,
                                      dilation=1)
        self.location_dense = LinearNorm(attention_n_filters, attention_dim,
                                         bias=False, w_init_gain='tanh')

    def forward(self, attention_weights_cat):
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention


class Attention(nn.Module):
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size):
        super(Attention, self).__init__()
        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim,
                                      bias=False, w_init_gain='tanh')
        self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False,
                                       w_init_gain='tanh')
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim)
        self.score_mask_value = -float("inf")

    def get_alignment_energies(self, query, processed_memory,
                               attention_weights_cat):
        """
        PARAMS
        ------
        query: decoder output (batch, n_mel_channels * n_frames_per_step)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)

        RETURNS
        -------
        alignment (batch, max_time)
        """

        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + processed_memory))

        energies = energies.squeeze(-1)
        return energies

    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """
        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat)

        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights


class Prenet(nn.Module):
    def __init__(self, in_dim, sizes, prenet_type='dropout'):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.prenet_type = prenet_type
        if self.prenet_type == 'dropout' or 'only_linear':
            self.layers = nn.ModuleList(
                [LinearNorm(in_size, out_size, bias=False)
                for (in_size, out_size) in zip(in_sizes, sizes)])
        elif self.prenet_type == 'batchnorm':
            self.layers = nn.ModuleList(
                [LinearBN(in_size, out_size, bias=False)
                for (in_size, out_size) in zip(in_sizes, sizes)])

    def forward(self, x):
        if self.prenet_type == 'dropout':
            for linear in self.layers:
                x = F.dropout(F.relu(linear(x)), p=0.5, training=True)
        elif self.prenet_type == 'only_linear' or self.prenet_type == 'batchnorm':
            for linear in self.layers:
                x = F.relu(linear(x))
        return x


class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self, hparams):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.n_mel_channels, hparams.postnet_embedding_dim,
                         kernel_size=hparams.postnet_kernel_size, stride=1,
                         padding=int((hparams.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(hparams.postnet_embedding_dim))
        )

        for i in range(1, hparams.postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(hparams.postnet_embedding_dim,
                             hparams.postnet_embedding_dim,
                             kernel_size=hparams.postnet_kernel_size, stride=1,
                             padding=int((hparams.postnet_kernel_size - 1) / 2),
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(hparams.postnet_embedding_dim))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.postnet_embedding_dim, hparams.n_mel_channels,
                         kernel_size=hparams.postnet_kernel_size, stride=1,
                         padding=int((hparams.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(hparams.n_mel_channels))
            )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)

        return x


class Encoder(nn.Module):
    """Encoder module:
        - Three 1-d convolution banks
        - Bidirectional LSTM
    """
    def __init__(self, hparams):
        super(Encoder, self).__init__()

        convolutions = []
        if hparams.speech_dewarping or hparams.naive_speech_autoencoder:
            start_dim = hparams.mel_embedding_dim
        elif hparams.text_finetune:
            start_dim = hparams.symbols_embedding_dim
        for idx in range(hparams.encoder_n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(start_dim if idx==0 else hparams.encoder_embedding_dim,
                         hparams.encoder_embedding_dim,
                         kernel_size=hparams.encoder_kernel_size, stride=1,
                         padding=int((hparams.encoder_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(hparams.encoder_embedding_dim))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.forward_cell = nn.LSTMCell(hparams.encoder_embedding_dim, int(hparams.encoder_embedding_dim / 2))
        self.backward_cell = nn.LSTMCell(hparams.encoder_embedding_dim, int(hparams.encoder_embedding_dim / 2))
        self.zoneout_rnn = ZoneoutRNN(self.forward_cell, self.backward_cell, (hparams.p_zoneout, hparams.p_zoneout))

    def forward(self, x):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)
        
        outputs = torch.zeros_like(x)
        forward_h, forward_c, backward_h, backward_c = torch.zeros(x.shape[0], x.shape[2]//2).to(x.device), torch.zeros(x.shape[0], x.shape[2]//2).to(x.device), torch.zeros(x.shape[0], x.shape[2]//2).to(x.device), torch.zeros(x.shape[0], x.shape[2]//2).to(x.device)
        forward_state = (forward_h, forward_c)
        backward_state = (backward_h, backward_c)
        max_time = x.shape[1]
        for i in range(max_time):
            forward_input = x[:, i, :]
            backward_input = x[:, max_time-(i+1), :]
            forward_output, backward_output, forward_new_state, backward_new_state = self.zoneout_rnn(forward_input, backward_input, forward_state, backward_state)
            forward_state = forward_new_state
            backward_state = backward_new_state
            outputs[:, i, :x.shape[2]//2] = forward_output
            outputs[:, max_time-(i+1), x.shape[2]//2:] = backward_output

        return outputs

    def inference(self, x):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        outputs = torch.zeros_like(x)
        forward_h, forward_c, backward_h, backward_c = torch.zeros(x.shape[0], x.shape[2]//2).to(x.device), torch.zeros(x.shape[0], x.shape[2]//2).to(x.device), torch.zeros(x.shape[0], x.shape[2]//2).to(x.device), torch.zeros(x.shape[0], x.shape[2]//2).to(x.device)
        forward_state = (forward_h, forward_c)
        backward_state = (backward_h, backward_c)
        max_time = x.shape[1]
        for i in range(max_time):
            forward_input = x[:, i, :]
            backward_input = x[:, max_time-(i+1), :]
            forward_output, backward_output, forward_new_state, backward_new_state = self.zoneout_rnn(forward_input, backward_input, forward_state, backward_state)
            forward_state = forward_new_state
            backward_state = backward_new_state
            outputs[:, i, :x.shape[2]//2] = forward_output
            outputs[:, max_time-(i+1), x.shape[2]//2:] = backward_output           
        return outputs


class Decoder(nn.Module):
    def __init__(self, hparams):
        super(Decoder, self).__init__()
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.encoder_embedding_dim = hparams.encoder_embedding_dim
        self.attention_rnn_dim = hparams.attention_rnn_dim
        self.decoder_rnn_dim = hparams.decoder_rnn_dim
        self.prenet_dim = hparams.prenet_dim
        self.max_decoder_steps = hparams.max_decoder_steps
        self.gate_threshold = hparams.gate_threshold
        self.p_attention_dropout = hparams.p_attention_dropout
        self.p_decoder_dropout = hparams.p_decoder_dropout
        self.hp = hparams
        if hparams.concat_speaker_embedding:
            self.encoder_embedding_dim = hparams.encoder_embedding_dim + hparams.speaker_embedding_dim
            hparams.encoder_embedding_dim = hparams.encoder_embedding_dim + hparams.speaker_embedding_dim

        self.prenet = Prenet(
            hparams.n_mel_channels * hparams.n_frames_per_step,
            [hparams.prenet_dim, hparams.prenet_dim], prenet_type=hparams.prenet_type)

        self.attention_rnn = nn.LSTMCell(
            hparams.prenet_dim + hparams.encoder_embedding_dim,
            hparams.attention_rnn_dim)

        self.attention_layer = Attention(
            hparams.attention_rnn_dim, hparams.encoder_embedding_dim,
            hparams.attention_dim, hparams.attention_location_n_filters,
            hparams.attention_location_kernel_size)

        self.decoder_rnn = nn.LSTMCell(
            hparams.attention_rnn_dim + hparams.encoder_embedding_dim,
            hparams.decoder_rnn_dim, 1)

        self.linear_projection = LinearNorm(
            hparams.decoder_rnn_dim + hparams.encoder_embedding_dim,
            hparams.n_mel_channels * hparams.n_frames_per_step)

        self.gate_layer = LinearNorm(
            hparams.decoder_rnn_dim + hparams.encoder_embedding_dim, 1,
            bias=True, w_init_gain='sigmoid')

    def get_go_frame(self, memory):
        """ Gets all zeros frames to use as first decoder input
        PARAMS
        ------
        memory: decoder outputs

        RETURNS
        -------
        decoder_input: all zeros frames
        """
        B = memory.size(0)
        decoder_input = Variable(memory.data.new(
            B, self.n_mel_channels * self.n_frames_per_step).zero_())
        return decoder_input

    def initialize_decoder_states(self, memory, mask):
        """ Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, attention context, stores memory
        and stores processed memory
        PARAMS
        ------
        memory: Encoder outputs
        mask: Mask for padded data if training, expects None for inference
        """
        B = memory.size(0)
        MAX_TIME = memory.size(1)

        self.attention_hidden = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())
        self.attention_cell = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())

        self.decoder_hidden = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())
        self.decoder_cell = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())

        self.attention_weights = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_weights_cum = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_context = Variable(memory.data.new(
            B, self.encoder_embedding_dim).zero_())

        self.memory = memory
        self.processed_memory = self.attention_layer.memory_layer(memory)
        self.mask = mask

    def parse_decoder_inputs(self, decoder_inputs):
        """ Prepares decoder inputs, i.e. mel outputs
        PARAMS
        ------
        decoder_inputs: inputs used for teacher-forced training, i.e. mel-specs

        RETURNS
        -------
        inputs: processed decoder inputs

        """
        # (B, n_mel_channels, T_out) -> (B, T_out, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(1, 2)
        decoder_inputs = decoder_inputs.reshape(
            decoder_inputs.size(0),
            int(decoder_inputs.size(1)/self.n_frames_per_step), -1)
        # (B, T_out, n_mel_channels) -> (T_out, B, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs

    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
        """ Prepares decoder outputs for output
        PARAMS
        ------
        mel_outputs:
        gate_outputs: gate output energies
        alignments:

        RETURNS
        -------
        mel_outputs:
        gate_outpust: gate output energies
        alignments:
        """
        # (T_out, B) -> (B, T_out)
        alignments = torch.stack(alignments).transpose(0, 1)
        # (T_out, B) -> (B, T_out)
        gate_outputs = torch.stack(gate_outputs).transpose(0, 1)
        gate_outputs = gate_outputs.contiguous()
        gate_outputs = gate_outputs.repeat_interleave(self.hp.n_frames_per_step, 1)
        # (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        # decouple frames per step
        mel_outputs = mel_outputs.view(
            mel_outputs.size(0), -1, self.n_mel_channels)
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)

        return mel_outputs, gate_outputs, alignments

    def decode(self, decoder_input):
        """ Decoder step using stored states, attention and memory
        PARAMS
        ------
        decoder_input: previous mel output

        RETURNS
        -------
        mel_output:
        gate_output: gate output energies
        attention_weights:
        """
        cell_input = torch.cat((decoder_input, self.attention_context), -1)
        attention_hidden_new, attention_cell_new = self.attention_rnn(
            cell_input, (self.attention_hidden, self.attention_cell)) # self.attention_hidden.shape = [B, attention_rnn_dim=1024]
        
        self.attention_hidden = (1 - self.hp.p_zoneout) * F.dropout(attention_hidden_new - self.attention_hidden, p=self.hp.p_zoneout, training=self.training) + self.attention_hidden
        self.attention_cell = (1 - self.hp.p_zoneout) * F.dropout(attention_cell_new - self.attention_cell, p=self.hp.p_zoneout, training=self.training) + self.attention_cell
        
        attention_weights_cat = torch.cat(
            (self.attention_weights.unsqueeze(1),
             self.attention_weights_cum.unsqueeze(1)), dim=1)
        self.attention_context, self.attention_weights = self.attention_layer(
            self.attention_hidden, self.memory, self.processed_memory,
            attention_weights_cat, self.mask)

        self.attention_weights_cum += self.attention_weights
        decoder_input = torch.cat(
            (self.attention_hidden, self.attention_context), -1)
        decoder_hidden_new, decoder_cell_new = self.decoder_rnn(
            decoder_input, (self.decoder_hidden, self.decoder_cell))
        
        self.decoder_hidden = (1 - self.hp.p_zoneout) * F.dropout(decoder_hidden_new - self.decoder_hidden, p=self.hp.p_zoneout, training=self.training) + self.decoder_hidden
        self.decoder_cell = (1 - self.hp.p_zoneout) * F.dropout(decoder_cell_new - self.decoder_cell, p=self.hp.p_zoneout, training=self.training) + self.decoder_cell

        decoder_hidden_attention_context = torch.cat(
            (self.decoder_hidden, self.attention_context), dim=1)
        decoder_output = self.linear_projection(
            decoder_hidden_attention_context)

        gate_prediction = self.gate_layer(decoder_hidden_attention_context)
        return decoder_output, gate_prediction, self.attention_weights
    
    def get_next_input(self, step, memory, decoder_inputs):
        if step == 0:
            return self.get_go_frame(memory)
        return decoder_inputs[step-1]
        
    def forward(self, memory, decoder_inputs, memory_lengths):
        """ Decoder forward pass for training
        PARAMS
        ------
        memory: Encoder outputs
        decoder_inputs: Decoder inputs for teacher forcing. i.e. mel-specs
        memory_lengths: Encoder output lengths for attention masking.

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)

        self.initialize_decoder_states(
            memory, mask=~get_mask_from_lengths(memory_lengths))
        mel_output = None
        mel_outputs, gate_outputs, alignments = [], [], []
        for step in range(decoder_inputs.size(0)):
            decoder_input = self.get_next_input(step, memory, decoder_inputs)
            decoder_input = self.prenet(decoder_input)
            mel_output, gate_output, attention_weights = self.decode(
                decoder_input)
            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output.squeeze(1)]
            alignments += [attention_weights]

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)

        return mel_outputs, gate_outputs, alignments

    def inference(self, memory):
        """ Decoder inference
        PARAMS
        ------
        memory: Encoder outputs

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        # print("[Decoder.inference] self.training", self.training)
        decoder_input = self.get_go_frame(memory)

        self.initialize_decoder_states(memory, mask=None)

        mel_outputs, gate_outputs, alignments = [], [], []
        while True:
            decoder_input = self.prenet(decoder_input)
            mel_output, gate_output, alignment = self.decode(decoder_input)

            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output]
            alignments += [alignment]

            if torch.sigmoid(gate_output.data) > self.gate_threshold:
                break
            elif self.hp.n_frames_per_step * len(mel_outputs) >= self.max_decoder_steps:
                print("Warning! Reached max decoder steps")
                break

            decoder_input = mel_output

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)

        return mel_outputs, gate_outputs, alignments


class Tacotron2(nn.Module):
    def __init__(self, hparams):
        super(Tacotron2, self).__init__()
        self.hp = hparams
        self.mask_padding = hparams.mask_padding
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        if hparams.iskorean:
            n_symbols = hparams.n_symbols_korean
        else:
            n_symbols = hparams.n_symbols
        self.embedding = nn.Embedding(n_symbols, hparams.symbols_embedding_dim)
        self.speaker_embedding = nn.Embedding(hparams.num_speaker, hparams.speaker_embedding_dim)

        std = sqrt(2.0 / (n_symbols + hparams.symbols_embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)
        self.encoder = Encoder(hparams)
        self.decoder = Decoder(hparams)
        self.postnet = Postnet(hparams)

        self.resized_mel_encoder = nn.Sequential(
                    ConvNorm(hparams.n_mel_channels,
                            hparams.mel_embedding_dim,
                            kernel_size=hparams.encoder_kernel_size, stride=1,
                            padding=int((hparams.encoder_kernel_size - 1) / 2),
                            dilation=1, w_init_gain='relu'),
                    nn.BatchNorm1d(hparams.mel_embedding_dim))
        
    def parse_batch(self, batch):
        text_padded, resized_mel_padded, input_lengths, mel_padded, gate_padded, output_lengths, speaker_idxs = batch
        if self.hp.text_finetune:
            text_padded = to_gpu(text_padded).long()
        if self.hp.speech_dewarping or self.hp.naive_speech_autoencoder:
            resized_mel_padded = to_gpu(resized_mel_padded).float()
        input_lengths = to_gpu(input_lengths).long()
        max_len = torch.max(input_lengths.data).item()
        mel_padded = to_gpu(mel_padded).float()
        gate_padded = to_gpu(gate_padded).float()
        output_lengths = to_gpu(output_lengths).long()
        speaker_idxs = to_gpu(speaker_idxs).long()

        return (
            (text_padded, resized_mel_padded, input_lengths, mel_padded, max_len, output_lengths, speaker_idxs),
            (mel_padded, gate_padded, resized_mel_padded))

    def parse_output(self, outputs, output_lengths=None):
        if self.mask_padding and output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths)
            mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
            if mask.size(2) % self.n_frames_per_step != 0 :
                to_append = torch.ones( mask.size(0), mask.size(1), (self.n_frames_per_step-mask.size(2)%self.n_frames_per_step) ).bool().to(mask.device)
                mask = torch.cat([mask, to_append], dim=-1)

            mask = mask.permute(1, 0, 2)
            outputs[0].data.masked_fill_(mask, 0.0)
            outputs[1].data.masked_fill_(mask, 0.0)
            outputs[2].data.masked_fill_(mask[:, 0, :], 1e3)  # gate energies

        return outputs

    def forward(self, inputs):
        text_inputs, resized_mel_inputs, text_lengths, mels, max_len, output_lengths, speaker_idxs = inputs
        text_lengths, output_lengths = text_lengths.data, output_lengths.data
        batch_size = mels.shape[0]
        
        if self.hp.text_finetune:
            embedded_inputs = self.embedding(text_inputs).transpose(1, 2)
        elif self.hp.speech_dewarping or self.hp.naive_speech_autoencoder:
            embedded_inputs = self.resized_mel_encoder(resized_mel_inputs)

        encoder_outputs = self.encoder(embedded_inputs)
        
        if self.hp.concat_speaker_embedding:
            speaker_embedded = self.speaker_embedding(speaker_idxs.unsqueeze(1)).repeat(1, encoder_outputs.shape[1], 1).float()
            encoder_outputs = torch.cat([encoder_outputs, speaker_embedded], dim = -1)

        mel_outputs, gate_outputs, alignments = self.decoder(encoder_outputs, mels, memory_lengths = text_lengths)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments],
            output_lengths)

    def inference(self, inputs, speaker_idx=0):
        if self.hp.text_finetune:
            embedded_inputs = self.embedding(inputs.long()).transpose(1, 2)
        if self.hp.speech_dewarping or self.hp.naive_speech_autoencoder:
            embedded_inputs = self.resized_mel_encoder(inputs)
        encoder_outputs = self.encoder.inference(embedded_inputs)
        
        if self.hp.concat_speaker_embedding:
            speaker_idxs = torch.Tensor([speaker_idx]).to(encoder_outputs.device).long()
            speaker_embedded = self.speaker_embedding(speaker_idxs.unsqueeze(1)).repeat(1, encoder_outputs.shape[1], 1).float()
            encoder_outputs = torch.cat([encoder_outputs, speaker_embedded], dim = -1)

        mel_outputs, gate_outputs, alignments = self.decoder.inference(
            encoder_outputs)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        outputs = self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments])

        return outputs
