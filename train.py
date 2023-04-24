import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import subprocess
import os
import time
import argparse
import math
from numpy import finfo

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from distributed import apply_gradient_allreduce

from model import Tacotron2
from data_utils import TextMelLoader, TextMelCollate
from loss_function import Tacotron2Loss
from logger import Tacotron2Logger
from hparams import get_hparams, add_hparams


def reduce_tensor(tensor, n_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= n_gpus
    return rt

def init_distributed(hparams, n_gpus, rank, group_name):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."
    print("Initializing Distributed")

    # Set cuda device so everything is done on the right GPU.
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # Initialize distributed communication
    dist.init_process_group(
        backend=hparams.dist_backend, init_method=hparams.dist_url,
        world_size=n_gpus, rank=rank, group_name=group_name)

    print("Done initializing distributed")

def prepare_dataloaders(hparams):
    # Get data, data loaders and collate function ready
    trainset = TextMelLoader(hparams.training_files, hparams)
    valset = TextMelLoader(hparams.validation_files, hparams)
    collate_fn = TextMelCollate(hparams.n_frames_per_step, hparams)

    if hparams.distributed_run:
        train_sampler = DistributedSampler(trainset)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_loader = DataLoader(trainset, num_workers=1, shuffle=shuffle,
                              sampler=train_sampler,
                              batch_size=hparams.batch_size, pin_memory=False,
                              drop_last=True, collate_fn=collate_fn)
    return train_loader, valset, collate_fn

def prepare_directories_and_logger(output_directory, log_directory, rank):
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        logger = Tacotron2Logger(os.path.join(output_directory, log_directory), hparams)
    else:
        logger = None
    return logger

def load_model(hparams):
    model = Tacotron2(hparams).cuda()
    if hparams.fp16_run:
        model.decoder.attention_layer.score_mask_value = finfo('float16').min

    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)

    return model

def k_not_in_ignore_layers(k, ignore_layers):
    for ignore_layer in ignore_layers:
        if ignore_layer in k:
            return False
    return True

def warm_start_model(checkpoint_path, model, ignore_layers):
    assert os.path.isfile(checkpoint_path)
    print("Warm starting model from checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model_dict = checkpoint_dict['state_dict']
    if len(ignore_layers) > 0:
        model_dict = {k: v for k, v in model_dict.items()
                      if k_not_in_ignore_layers(k, ignore_layers)}
        dummy_dict = model.state_dict()
        dummy_dict.update(model_dict)
        model_dict = dummy_dict
    model.load_state_dict(model_dict, strict=False)
    return model

def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    try:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    except:
        pass
    if hparams.load_scheduler:
        scheduler.load_state_dict(checkpoint_dict['scheduler'])
    learning_rate = checkpoint_dict['learning_rate']
    iteration = checkpoint_dict['iteration']
    print("Loaded checkpoint '{}' from iteration {}" .format(
        checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration, scheduler

def save_checkpoint(model, optimizer, learning_rate, iteration, filepath, scheduler):
    print("Saving model and optimizer state at iteration {} to {}".format(
        iteration, filepath))
    torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler':scheduler.state_dict() if scheduler is not None else None,
                'learning_rate': learning_rate}, filepath)

# With teacher-forcing
def validate(model, criterion, valset, iteration, batch_size, n_gpus, collate_fn, logger, distributed_run, rank):
    model.eval()
    with torch.no_grad():
        val_sampler = DistributedSampler(valset) if distributed_run else None
        val_loader = DataLoader(valset, sampler=val_sampler, num_workers=1,
                                shuffle=False, batch_size=batch_size,
                                pin_memory=False, collate_fn=collate_fn)

        val_loss = 0.0
        val_mel_loss = 0.0
        val_gate_loss = 0.0
        if iteration > hparams.segaug_end_step:
            val_loader.dataset.hp.segaug = False

        for i, batch in enumerate(val_loader):
            batch = batch[:-1]

            x, y = model.parse_batch(batch)
            y_pred = model(x)

            mel_loss, gate_loss = criterion(y_pred, y)        

            loss = mel_loss + hparams.gate_loss_weight * gate_loss
            mel_loss, gate_loss = mel_loss.item(), gate_loss.item()
            
            if distributed_run:
                reduced_val_loss = reduce_tensor(loss.data, n_gpus).item()
            else:
                reduced_val_loss = loss.item()
            val_loss += reduced_val_loss
            val_mel_loss += mel_loss
            val_gate_loss += gate_loss
        
        val_loss = val_loss / (i + 1)
        val_mel_loss = val_mel_loss / (i + 1)
        val_gate_loss = val_gate_loss / (i + 1)
        
    model.train()
    if rank == 0:
        print(f"Validation loss {iteration}: {val_loss:9f} mel_loss {val_mel_loss:.6f} gate_loss {val_gate_loss:.7f}")
        logger.log_validation(val_loss, mel_loss, gate_loss, model, y, y_pred, iteration)

def train(output_directory, log_directory, checkpoint_path, warm_start, n_gpus,
          rank, group_name, hparams):

    if hparams.distributed_run:
        init_distributed(hparams, n_gpus, rank, group_name)

    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)

    model = load_model(hparams)
    learning_rate = hparams.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=hparams.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                            lr_lambda=lambda step: 1 if step < hparams.start_decay else max(hparams.final_lr/hparams.learning_rate, 1 * (hparams.decay_rate ** ((step-hparams.start_decay)/hparams.decay_steps))),
                                            last_epoch=-1,
                                            verbose=False) if hparams.use_lr_schedule else None

    if hparams.fp16_run:
        from apex import amp
        model, optimizer = amp.initialize(
            model, optimizer, opt_level='O2')

    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)

    logger = prepare_directories_and_logger(
        output_directory, log_directory, rank)

    train_loader, valset, collate_fn = prepare_dataloaders(hparams)

    # Load checkpoint if one exists
    iteration = 0
    epoch_offset = 0
    if checkpoint_path is not None:
        if warm_start:
            model = warm_start_model(
                checkpoint_path, model, hparams.ignore_layers)
        else:
            model, optimizer, _learning_rate, iteration, scheduler = load_checkpoint(
                checkpoint_path, model, optimizer, scheduler)
            if hparams.use_saved_learning_rate:
                learning_rate = _learning_rate
            iteration += 1  # next iteration is iteration + 1
            epoch_offset = max(0, int(iteration / len(train_loader)))

    model.train()
    is_overflow = False
    # ================ MAIN TRAINNIG LOOP! ===================
    for epoch in range(epoch_offset, hparams.epochs):
        print("Epoch: {}".format(epoch))
        for i, batch in enumerate(train_loader):
            start = time.perf_counter()
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
            model.zero_grad()
            
            batch, wav_paths = batch[:-1], batch[-1]
            x, y = model.parse_batch(batch)
            if iteration > hparams.segaug_end_step:
                train_loader.dataset.hp.segaug = False

            y_pred = model(x)
            mel_loss, gate_loss = Tacotron2Loss()(y_pred, y)
            loss = mel_loss + hparams.gate_loss_weight * gate_loss

            if hparams.distributed_run:
                reduced_loss = reduce_tensor(loss.data, n_gpus).item()
            else:
                reduced_loss = loss.item()
            if hparams.fp16_run:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if hparams.fp16_run:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    amp.master_params(optimizer), hparams.grad_clip_thresh)
                is_overflow = math.isnan(grad_norm)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), hparams.grad_clip_thresh)

            optimizer.step()
            if hparams.use_lr_schedule:
                scheduler.step()
                learning_rate = scheduler.get_last_lr()[0]
            else:
                learning_rate = optimizer.param_groups[0]["lr"]
            mel_loss, gate_loss = mel_loss.item(), gate_loss.item()

            if not is_overflow and rank == 0:
                duration = time.perf_counter() - start
                print(f"Iteration {iteration} Train loss {reduced_loss:.6f} mel_loss {mel_loss:.6f} gate_loss {gate_loss:.7f} Grad Norm {grad_norm:.6f} lr {learning_rate} {duration:.2f}s/it")
                logger.log_training(
                    reduced_loss, mel_loss, gate_loss, grad_norm, learning_rate, duration, iteration)

            if not is_overflow and (iteration % hparams.iters_per_checkpoint == 0):
                validate(model, Tacotron2Loss(), valset, iteration,
                         hparams.batch_size, n_gpus, collate_fn, logger,
                         hparams.distributed_run, rank)
                if rank == 0:
                    checkpoint_path = os.path.join(
                        output_directory, "checkpoint_{}".format(iteration))
                    save_checkpoint(model, optimizer, learning_rate, iteration,
                                    checkpoint_path, scheduler if hparams.use_lr_schedule else None)
            iteration += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_directory', type=str,
                        help='directory to save checkpoints')
    parser.add_argument('-l', '--log_directory', type=str,
                        help='directory to save tensorboard logs', default = 'runs')
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        required=False, help='checkpoint path')
    parser.add_argument('--warm_start', action='store_true',
                        help='load model weights only, ignore specified layers')
    parser.add_argument('--n_gpus', type=int, default=1,
                        required=False, help='number of gpus')
    parser.add_argument('--rank', type=int, default=0,
                        required=False, help='rank of current gpu')
    parser.add_argument('--group_name', type=str, default='group_name',
                        required=False, help='Distributed group name')
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')

    args = parser.parse_args()
    add_hparams(parser)
    args = parser.parse_args()
    hparams = get_hparams(args, parser)

    os.makedirs(args.output_directory, exist_ok=True)
    subprocess.call(f"cp hparams.py {args.output_directory}", shell = True)
    
    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

    print("FP16 Run:", hparams.fp16_run)
    print("Dynamic Loss Scaling:", hparams.dynamic_loss_scaling)
    print("Distributed Run:", hparams.distributed_run)
    print("cuDNN Enabled:", hparams.cudnn_enabled)
    print("cuDNN Benchmark:", hparams.cudnn_benchmark)

    train(args.output_directory, args.log_directory, args.checkpoint_path,
          args.warm_start, args.n_gpus, args.rank, args.group_name, hparams)
