import torch

sd = torch.load("model.ckpt")
unet_sd = {k[22:]: v for k, v in sd.items() if k[:21]=='model.diffusion_model'}
vq_sd = {k[18:]: v for k, v in sd.items() if k[:17]=='first_stage_model'}
cond_sd = {k[17:]: v for k, v in sd.items() if k[:16]=='cond_stage_model'}

torch.save(unet_sd, 'unet.ckpt')
torch.save(vq_sd, 'vqvae.ckpt')
torch.save(cond_sd, 'clip.ckpt')
