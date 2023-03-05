import os
import torch
import argparse
import json
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
from data import get_dataset
from train_utils import get_models, get_DDPM, get_logger_and_callbacks

if __name__ == '__main__':
    seed_everything(42)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config', type=str, 
        default='config/train.json')
    parser.add_argument(
        '-r', '--resume', action="store_true"
    )
    parser.add_argument(
        '-n', '--nnode', type=int, default=1
    )

    ''' parser configs '''
    args_raw = parser.parse_args()
    with open(args_raw.config, 'r') as IN:
        args = json.load(IN)
    args['resume'] = args_raw.resume
    args['nnode'] = args_raw.nnode
    expt_name = args['expt_name']
    expt_dir = args['expt_dir']
    expt_path = os.path.join(expt_dir, expt_name)
    os.makedirs(expt_path, exist_ok=True)

    '''1. create denoising model'''
    models = get_models(args)

    diffusion_configs = args['diffusion']
    ddpm_model = get_DDPM(
        diffusion_configs=diffusion_configs,
        log_args=args,
        **models
    )

    '''2. dataset and dataloader'''
    data_args = args['data']
    train_set, val_set = get_dataset(**data_args)
    train_loader = DataLoader(
        train_set, batch_size=data_args['batch_size'], shuffle=True,
        num_workers=4*len(args['trainer_args']['devices']), pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=data_args['val_batch_size'],
        num_workers=len(args['trainer_args']['devices']), pin_memory=True
    )
    '''3. create callbacks'''
    wandb_logger, callbacks = get_logger_and_callbacks(expt_name, expt_path, args)

    '''4. trainer'''
    trainer_args = {
        "max_epochs": 1000,
        "accelerator": "gpu",
        "devices": [0],
        "limit_val_batches": 1,
        "strategy": "ddp",
        "check_val_every_n_epoch": 1,
        "num_nodes": args['nnode']
        # "benchmark" :True
    }
    config_trainer_args = args['trainer_args'] if args.get('trainer_args') is not None else {}
    trainer_args.update(config_trainer_args)
    print(f'Training args are {trainer_args}')
    trainer = Trainer(
        logger = wandb_logger,
        callbacks = callbacks,
        **trainer_args
    )
    '''5. start training'''
    if args['resume']:
        print('INFO: Try to resume from checkpoint')
        ckpt_path = os.path.join(expt_path, 'latest.ckpt')
        if os.path.exists(ckpt_path):
            print(f'INFO: Found checkpoint {ckpt_path}')
            # ckpt = torch.load(ckpt_path, map_location='cpu')['state_dict']
            # ddpm_model.load_state_dict(ckpt)
        else:
            ckpt_path = None
    else:
        ckpt_path = None
    trainer.fit(
        ddpm_model, train_loader, val_loader,
        ckpt_path=ckpt_path
    )
        