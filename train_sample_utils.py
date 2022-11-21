from functools import partial
from modules.openai_unet.openaimodel import UNetModel as OpenAIUNet
# from modules.vae.vae import BetaVAE
from pytorch_lightning.loggers import WandbLogger
from callbacks import get_epoch_checkpoint, get_latest_checkpoint, get_iteration_checkpoint
from model_utils import instantiate_from_config, get_obj_from_str

def get_models(args):
    denoise_model = args['denoising_model']['model']
    denoise_args = args['denoising_model']['model_args']
    denoise_fn = instantiate_from_config({
        'target': denoise_model,
        'params': denoise_args
    })
    model_dict = {
        'denoise_fn': denoise_fn,
    }

    if args.get('vqvae_model'):
        vq_model = args['vqvae_model']['model']
        vq_args = args['vqvae_model']['model_args']
        vqvae_fn = instantiate_from_config({
            'target': vq_model,
            'params': vq_args
        })

        model_dict['vqvae_fn'] = vqvae_fn

    if args.get('text_model'):
        text_model = args['text_model']['model']
        text_args = args['text_model']['model_args']
        text_fn = instantiate_from_config({
            'target': text_model,
            'params': text_args
        })

        model_dict['text_fn'] = text_fn

    return model_dict

def get_DDPM(diffusion_configs, log_args={}, **models):
    diffusion_model_class = diffusion_configs['model']
    diffusion_args = diffusion_configs['model_args']
    DDPM_model = get_obj_from_str(diffusion_model_class)
    ddpm_model = DDPM_model(
        log_args=log_args,
        **models,
        **diffusion_args
    )
    return ddpm_model


def get_logger_and_callbacks(expt_name, expt_path, args):
    callbacks = []
    # 3.1 checkpoint callbacks
    save_model_config = args.get('save_model_config', {})
    epoch_checkpoint = get_epoch_checkpoint(expt_path, **save_model_config)
    latest_checkpoint = get_latest_checkpoint(expt_path)
    callbacks.append(epoch_checkpoint)
    callbacks.append(latest_checkpoint)

    # 3.2 wandb logger
    wandb_logger = WandbLogger(
        project=expt_name,
    )
    iteration_callbacks = args.get('iteration_callbacks')
    if iteration_callbacks:
        callbacks.append(get_iteration_checkpoint(expt_path))
    config_callbacks = args.get('callbacks')
    if config_callbacks is not None:
        for callback in config_callbacks:
            print(f'Initiate callback {callback}')
            callbacks.append(
                get_obj_from_str(callback)(
                    wandb_logger=wandb_logger,
                    max_num_images=8
                )
            )
    else:
        from callbacks import WandBImageLogger
        print(f'INFO: got {expt_name}, will use default image logger')
        wandb_callback = WandBImageLogger(
            wandb_logger=wandb_logger,
            max_num_images=8
        )
        callbacks.append(wandb_callback)

    return wandb_logger, callbacks
