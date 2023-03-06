import torch
from safetensors import safe_open

def load_safetensors(file_path):
    tensors = {}
    with safe_open(file_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors

sd = load_safetensors("counterfeitV25Pruned.safetensors")
unet_sd = {k[22:]: v for k, v in sd.items() if k[:21]=='model.diffusion_model'}
vq_sd = {k[18:]: v for k, v in sd.items() if k[:17]=='first_stage_model'}
cond_sd = {k[17:]: v for k, v in sd.items() if k[:16]=='cond_stage_model'}

torch.save(unet_sd, 'unet.ckpt')
torch.save(vq_sd, 'vqvae.ckpt')
torch.save(cond_sd, 'clip.ckpt')
