from text import symbols
from text_kr import symbols as symbols_korean
import argparse


defaults = {
    
    # Training Parameters
    'epochs':500000, # Total Epochs to run. Set to a large enough value.
    'iters_per_checkpoint':2500, # The number of iterations between saving checkpoints.
    'training_files':'filelists/libritts/train.txt', # Filelists for training data. For the specific format of each line, follow the example filelists.
    'validation_files':'filelists/libritts/val.txt', # Filelists for validation data.
    'seed':1234, # Random seed.
    'dynamic_loss_scaling':True,
    'fp16_run':False,
    'distributed_run':False,
    'dist_backend':"nccl",
    'dist_url':"tcp'://localhost':54321",
    'cudnn_enabled':True,
    'cudnn_benchmark':False,
    'ignore_layers':"['embedding.weight']", # When loading a checkpoint, model layers with a name that contains at least one string in this list will be initialized randomly. Only used when the --warm_start argument is used when running train.py.

    # Scenario setting Paramters
    # Within 'text_finetune', 'speech_dewarping', 'naive_speech_autoencoder', only one of these three options should be True.
    # Each being True respectively represent the text-fine-tuning step, speech-dewarping pre-training step, the Naive pre-training scheme in our paper.
    'text_finetune':False, 
    'speech_dewarping':True,
    'naive_speech_autoencoder':False,
    
    'iskorean':False, # This option matters only when text_finetune is True. It denotes whether the fine-tuning text is Korean or not. This repository only supports English and Korean.
    'naive_resize_factor': 1/6, # Refer to line14 of Section2.2 in our paper.
    'concat_speaker_embedding':True, # Whether to concatenate speaker embeddings to the Tacotron encoder's output.
    'num_speaker': 247, # Number of speakers. When using the LibriTTS train-clean-100 split, it is 247. When text-fine-tuning with a single speaker, set to 1.
    'speaker_embedding_dim':32, # The dimension of the speaker embeddings.

    'segaug':False, # Whether to use the proposed SegAug augmentation during text-fine tuning. In the pre-training scenarios, this option is not used.
    'segaug_end_step':25000, # The iteration to start the cool-down. Refer to line10 of the Data Augmentation paragraph in Section2.3 in our paper.

    # Audio Parameters
    'max_wav_value':32768.0,
    'sampling_rate':16000,
    'filter_length':2048,
    'hop_length':200,
    'win_length':800,
    'griffin_lim_iters':60,
    'power':1.5,

    'n_mel_channels':80,
    'mel_fmin':95.0,
    'mel_fmax':7600.0,
    'use_preemphasis':True,
    'preemphasis':0.97,
    'use_rescale':True,
    'rescaling_max':0.999,

    'use_bshall_mel':True,
    'magnitude_power':2.0,
    'ref_level_db':20,
    'min_level_db':-100,
    'max_abs_value':4.0,

    'allow_clipping_in_normalization':False,
    'symmetric_mels':True,
    
    # Model Parameters
    'n_symbols':len(symbols),
    'n_symbols_korean':len(symbols_korean),
    'symbols_embedding_dim':512,
    'mel_embedding_dim':512,
    'mel_resize_step_num':1,

    # Encoder parameters
    'encoder_kernel_size':5,
    'encoder_n_convolutions':3,
    'encoder_embedding_dim':512,

    # Decoder parameters
    'n_frames_per_step':2,
    'decoder_rnn_dim':1024,
    'prenet_dim':256,
    'max_decoder_steps':1000,
    'gate_threshold':0.05,
    'p_attention_dropout':0.1,
    'p_decoder_dropout':0.1,
    'p_zoneout':0.1,
    'prenet_type':'dropout',  # 'dropout', 'only_linear', 'batchnorm'

    # Attention parameters
    'attention_rnn_dim':1024,
    'attention_dim':128,

    # Location Layer parameters
    'attention_location_n_filters':32,
    'attention_location_kernel_size':31,

    # Mel-post processing network parameters
    'postnet_embedding_dim':512,
    'postnet_kernel_size':5,
    'postnet_n_convolutions':5,

    # Optimization Parameters
    'use_saved_learning_rate':False, # Whether to load the learning rate when loading a checkpoint.
    'learning_rate':1e-3, # Initial learning rate. Used when use_saved_learning_rate is False.
    'weight_decay':1e-6,
    'grad_clip_thresh':1.0,
    'batch_size':16,
    'mask_padding':True,  # set model's padded outputs to padded values
    'gate_loss_weight':1.0,
    
    'use_lr_schedule':False, # If True, a learning rate schedule is used. Refer to line186 in train.py.
    'load_scheduler':False, # Whether to load the when loading a checkpoint.
    'final_lr':1e-4,
    'start_decay':20000,
    'decay_steps':8000,
    'decay_rate':0.5,
}

def get_hparams(args, parser):
    arg_groups = {}
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        arg_groups[group.title] = argparse.Namespace(**group_dict)

    hparams = arg_groups['hparams']
    hparams.ignore_layers = eval(hparams.ignore_layers)

    return hparams

def add_hparams(parser):
    """Create model hyperparameters. Parse nondefault from given string."""

    hparams_group = parser.add_argument_group('hparams')
    for key, default in defaults.items():
        hparams_group.add_argument("--{}".format(key), type=type(default), default=default)
        
if __name__ == "__main__":
    print(len(symbols))