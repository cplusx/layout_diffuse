import os
import torch
import argparse
import json
from pytorch_lightning import Trainer
from train_sample_utils import get_models, get_DDPM
from test_sample_utils import load_model_weights

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config', type=str, 
        default='config/train.json')
    parser.add_argument(
        '-n', '--num_repeat', type=int, 
        default=1, help='the number of images for each condition')
    parser.add_argument(
        '-e', '--epoch', type=int, 
        default=None, help='which epoch to evaluate, if None, will use the latest')
    parser.add_argument(
        '--nnode', type=int, default=1
    )
    parser.add_argument(
        '--model_path', type=str,
        default=None, help='model path for generating layout diffuse, if not provided, will use the latest.ckpt')

    ''' parser configs '''
    args_raw = parser.parse_args()
    with open(args_raw.config, 'r') as IN:
        args = json.load(IN)
    args.update(vars(args_raw))
    # args['gpu_ids'] = [0] # DEBUG
    expt_name = args['expt_name']
    expt_dir = args['expt_dir']
    expt_path = os.path.join(expt_dir, expt_name)
    os.makedirs(expt_path, exist_ok=True)

    '''1. create denoising model'''
    denoise_args = args['denoising_model']['model_args']
    models = get_models(args)

    diffusion_configs = args['diffusion']
    ddpm_model = get_DDPM(
        diffusion_configs=diffusion_configs,
        log_args=args,
        **models
    )

    '''2. create a dataloader which generates'''
    from test_sample_utils import get_test_dataset, get_test_callbacks
    test_dataset, test_loader = get_test_dataset(args)

    '''3. callbacks'''
    callbacks = get_test_callbacks(args, expt_path)

    '''4. load checkpoint'''
    print('INFO: loading checkpoint')
    if args['model_path'] is not None:
        ckpt_path = args['model_path']
    else:
        expt_path = os.path.join(args['expt_dir'], args['expt_name'])
        if args['epoch'] is None:
            ckpt_to_use = 'latest.ckpt'
        else:
            ckpt_to_use = f'epoch={args["epoch"]:04d}.ckpt'
        ckpt_path = os.path.join(expt_path, ckpt_to_use)
    print(ckpt_path)
    if os.path.exists(ckpt_path):
        print(f'INFO: Found checkpoint {ckpt_path}')
        # ckpt = torch.load(ckpt_path, map_location='cpu')['state_dict']
        ''' DEBUG '''
        # ckpt_denoise_fn = {k.replace('denoise_fn.', ''): v for k, v in ckpt.items() if 'denoise_fn' in k}
        # ddpm_model.denoise_fn.load_state_dict(ckpt_denoise_fn)
        # ddpm_model.load_state_dict(ckpt)
    else:
        ckpt_path = None
        raise RuntimeError('Cannot do inference without pretrained checkpoint')

    '''5. trianer'''
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
        callbacks = callbacks,
        **trainer_args
    )
        
    '''6. start sampling'''
    '''use trainer for sampling, you need a image saver callback to save images, useful for generate many images'''
    num_loop = args['num_repeat']
    for _ in range(num_loop):
        # trainer.test(ddpm_model, test_loader) # DEBUG
        trainer.test(ddpm_model, test_loader, ckpt_path=ckpt_path)
