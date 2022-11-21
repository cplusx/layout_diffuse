from functools import partial
from modules.unet.unet import UNet
from modules.unet.partial_unet import PartialUNet
from modules.openai_unet.openaimodel import UNetModel as OpenAIUNet
# from modules.vae.vae import BetaVAE
from pytorch_lightning.loggers import WandbLogger
from callbacks import get_epoch_checkpoint, get_latest_checkpoint, get_iteration_checkpoint
from model_utils import instantiate_from_config, get_obj_from_str

def get_models(args):
    denoise_model = args['denoising_model']['model']
    denoise_args = args['denoising_model']['model_args']
    if '.' in denoise_model:
        denoise_fn = instantiate_from_config({
            'target': denoise_model,
            'params': denoise_args
        })
    elif denoise_model == 'UNet':
        denoise_fn = UNet(**denoise_args)
    elif denoise_model == 'PartialUNet':
        denoise_fn = PartialUNet(**denoise_args)
    elif denoise_model == 'OpenAIUNet':
        denoise_fn = OpenAIUNet(**denoise_args)
    model_dict = {
        'denoise_fn': denoise_fn,
    }

    if args.get('vqvae_model'):
        vq_model = args['vqvae_model']['model']
        vq_args = args['vqvae_model']['model_args']
        if '.' in vq_model:
            vqvae_fn = instantiate_from_config({
                'target': vq_model,
                'params': vq_args
            })
        else:
            from modules.vqvae.autoencoder import VQModelInterface
            vqvae_args = args['vqvae_model']['model_args']
            vqvae_fn = VQModelInterface(**vqvae_args)

        model_dict['vqvae_fn'] = vqvae_fn

    if args.get('text_model'):
        text_model = args['text_model']['model']
        text_args = args['text_model']['model_args']
        text_fn = instantiate_from_config({
            'target': text_model,
            'params': text_args
        })

        model_dict['text_fn'] = text_fn

    if args.get('perceiver_model'):
        from modules.perceiver.perceiver import ImagePerceiver
        perceiver_model = args['perceiver_model']['model']
        perceiver_args = args['perceiver_model']['model_args']
        if perceiver_model == 'ImagePerceiver':
            perceiver_fn = ImagePerceiver(**perceiver_args)
        else:
            raise NotImplementedError

        model_dict['perceiver_fn'] = perceiver_fn
        
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

'''
    if 'palette' in expt_name:
        from callbacks.palette.palette_wandb import PaletteWandBImageLogger as WandBImageLogger
    elif expt_name in [
        'celeb_mask_joint', 
        'celeb_mask_joint_pred_y0',
        'celeb_mask_joint_dynamic_bg_loss'
    ]:
        from callbacks.celeb_mask.celeb_mask_wandb import CelebMaskWandBImageLogger as WandBImageLogger
    elif expt_name in [
        'celeb_mask_joint_mask_embedding',
        'celeb_mask_joint_mask_embedding_ams',
        'celeb_mask_joint_mask_embedding_ddim',
    ]:
        from callbacks.celeb_mask.celeb_mask_wandb import CelebMaskEmbeddingWandBImageLogger as WandBImageLogger
'''
# def get_DDPM_old(diffusion_args, expt_name, log_args={}, **models):
#     '''import DDPM or improved DDPM (DDIM)'''
#     if expt_name == 'mnist':
#         from DDPM.DDPM_MNIST import DDPM_MNIST
#         DDPM_model = DDPM_MNIST
#     elif expt_name == 'cifar_ldm':
#         from DDIM_ldm.DDIM_ldm_CIFAR import DDIM_LDM_CIFAR
#         DDPM_model = DDIM_LDM_CIFAR
#     elif expt_name == 'celeb_mask':
#         from DDPM.DDPM_celeb_mask import DDPMCelebMask
#         DDPM_model = DDPMCelebMask
#     elif 'palette' in expt_name:
#         from DDPM.DDPM_Palette import DDPMPalette
#         DDPM_model = DDPMPalette
#     elif expt_name in [
#         'celeb_mask_joint', 
#         'celeb_mask_joint_pred_y0',
#         'celeb_mask_joint_dynamic_bg_loss'
#     ]:
#         from DDPM.DDPM_joint import DDPMJointCelebMask, joint_loss
#         DDPM_model = DDPMJointCelebMask
#         if diffusion_args['loss_fn'] == 'joint':
#             diffusion_args['loss_fn'] = joint_loss
#         else:
#             raise RuntimeError
#     elif expt_name in [
#         'celeb_mask_joint_mask_embedding',
#         'celeb_mask_joint_mask_embedding_ams',
#     ]:
#         from DDPM.DDPM_joint import DDPMJointCelebMaskMapping, joint_mask_embedding_loss
#         DDPM_model = DDPMJointCelebMaskMapping
#         if isinstance(diffusion_args['loss_fn'], dict):
#             diffusion_args['loss_fn'] = partial(
#                 joint_mask_embedding_loss, **diffusion_args['loss_fn']
#             )
#     elif expt_name in [
#         'celeb_mask_joint_mask_embedding_ddim',
#     ]:
#         from DDIM.DDIM_joint import DDIMJointCelebMaskMapping
#         from DDIM.schedule_sampler import create_named_schedule_sampler
#         DDPM_model = DDIMJointCelebMaskMapping
#         diffusion_args['schedule_sampler'] = create_named_schedule_sampler(
#             name=diffusion_args['schedule_sampler'],
#             n_timesteps=diffusion_args['beta_schedule_args']['n_timestep']
#         )
#     elif expt_name in [
#         'celeb_mask_partial_attention',
#     ]:
#         from DDIM.DDIM_celeb_mask import DDIMCelebMaskPartialAttention
#         from DDIM.schedule_sampler import create_named_schedule_sampler
#         DDPM_model = DDIMCelebMaskPartialAttention
#         diffusion_args['schedule_sampler'] = create_named_schedule_sampler(
#             name=diffusion_args['schedule_sampler'],
#             n_timesteps=diffusion_args['beta_schedule_args']['n_timestep']
#         )
#     elif expt_name in [
#         'coco_layout_partial_attention',
#     ]:
#         from DDIM.DDIM_coco_layout import DDIMCocoLayoutPartialAttention
#         from DDIM.schedule_sampler import create_named_schedule_sampler
#         DDPM_model = DDIMCocoLayoutPartialAttention
#         diffusion_args['schedule_sampler'] = create_named_schedule_sampler(
#             name=diffusion_args['schedule_sampler'],
#             n_timesteps=diffusion_args['beta_schedule_args']['n_timestep']
#         )
#     elif expt_name in [
#         'coco_perceiver_partial_attention',
#     ]:
#         from DDIM.DDIM_coco_perceiver import DDIMCocoPerceiverPartialAttention
#         from DDIM.schedule_sampler import create_named_schedule_sampler
#         DDPM_model = DDIMCocoPerceiverPartialAttention
#         diffusion_args['schedule_sampler'] = create_named_schedule_sampler(
#             name=diffusion_args['schedule_sampler'],
#             n_timesteps=diffusion_args['beta_schedule_args']['n_timestep']
#         )
#     elif expt_name == 'mnist_ddim':
#         from DDIM.DDIM_MNIST import DDIM_MNIST
#         from DDIM.schedule_sampler import create_named_schedule_sampler
#         DDPM_model = DDIM_MNIST
#         diffusion_args['schedule_sampler'] = create_named_schedule_sampler(
#             name=diffusion_args['schedule_sampler'],
#             n_timesteps=diffusion_args['beta_schedule_args']['n_timestep']
#         )
#     elif expt_name in ['cifar_ddim']:
#         from DDIM.DDIM_CIFAR import DDIM_CIFAR
#         from DDIM.schedule_sampler import create_named_schedule_sampler
#         DDPM_model = DDIM_CIFAR
#         diffusion_args['schedule_sampler'] = create_named_schedule_sampler(
#             name=diffusion_args['schedule_sampler'],
#             n_timesteps=diffusion_args['beta_schedule_args']['n_timestep']
#         )
#     elif expt_name in [
#         'cifar_ddim_vae_v1',
#         'cifar_ddim_vae_v2',
#         'cifar_ddim_vae_v3',
#         'cifar_ddim_vae_v4',
#     ]:
#         from DDIM.DDIM_CIFAR import DDIMVAE_CIFAR
#         from DDIM.schedule_sampler import create_named_schedule_sampler
#         DDPM_model = DDIMVAE_CIFAR
#         diffusion_args['schedule_sampler'] = create_named_schedule_sampler(
#             name=diffusion_args['schedule_sampler'],
#             n_timesteps=diffusion_args['beta_schedule_args']['n_timestep']
#         )
#     elif expt_name == "celebahq_pretrained_ldm":
#         from DDIM_ldm.DDIM_ldm_celeb import DDIM_LDM_pretrained_celeb
#         DDPM_model = DDIM_LDM_pretrained_celeb
#     else:
#         from DDPM.DDPM import DDPMTraining
#         DDPM_model = DDPMTraining
#     ddpm_model = DDPM_model(
#         log_args=log_args,
#         **models,
#         **diffusion_args
#     )
#     return ddpm_model