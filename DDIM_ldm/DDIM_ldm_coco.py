import torch
from .DDIM_ldm import DDIM_LDM_VQVAETraining, DDIM_LDM_Text_VQVAETraining
from model_utils import default
from einops import repeat

class DDIM_LDM_pretrained_COCO(DDIM_LDM_VQVAETraining):
    def initialize_unet(self, unet_init_weights):
        if unet_init_weights is not None:
            print(f'INFO: initialize denoising UNet from {unet_init_weights}, NOTE: without partial attention layers')
            model_sd = torch.load(unet_init_weights)
            self_model_sd = self.denoise_fn.state_dict()
            self_model_params = list(self.denoise_fn.named_parameters())
            self_model_k = list(map(lambda x: x[0], self_model_params))
            self.params_not_pretrained = []
            self.params_pretrained = []
            for k_idx in range(len(self_model_k)):
                this_k = self_model_k[k_idx]
                if this_k not in model_sd:
                    if this_k=='output_blocks.5.3.conv.weight':
                        self_model_sd[this_k] = model_sd['output_blocks.5.2.conv.weight']
                        self.params_pretrained.append(self_model_params[k_idx][1])
                    elif this_k=='output_blocks.5.3.conv.bias':
                        self_model_sd[this_k] = model_sd['output_blocks.5.2.conv.bias']
                        self.params_pretrained.append(self_model_params[k_idx][1])
                    elif this_k=='output_blocks.8.3.conv.weight':
                        self_model_sd[this_k] = model_sd['output_blocks.8.2.conv.weight']
                        self.params_pretrained.append(self_model_params[k_idx][1])
                    elif this_k=='output_blocks.8.3.conv.bias':
                        self_model_sd[this_k] = model_sd['output_blocks.8.2.conv.bias']
                        self.params_pretrained.append(self_model_params[k_idx][1])
                    else:
                        self.params_not_pretrained.append(self_model_params[k_idx][1])
                        continue
                elif (self_model_sd[this_k].shape == model_sd[this_k].shape) or (self_model_sd[this_k].shape == model_sd[this_k].squeeze(-1).shape):
                    self_model_sd[this_k] = model_sd[this_k]
                    self.params_pretrained.append(self_model_params[k_idx][1])
                else:
                    self.params_not_pretrained.append(self_model_params[k_idx][1])
                    continue

            self.denoise_fn.load_state_dict(self_model_sd)

    def configure_optimizers(self):
        if self.freeze_pretrained_weights:
            assert hasattr(self, 'params_not_pretrained')
            print('INFO: pretrained weights are not trainable')
            params = self.params_not_pretrained
        else:
            params = list(self.denoise_fn.parameters())
        if self.learn_logvar:
            params = params + [self.logvar]
        optimizer = torch.optim.Adam(params, **self.optim_args)
        return optimizer

    @torch.no_grad()
    def fast_sampling(self, noise, model_kwargs={}):
        y_0, y_t_hist = super().fast_sampling(
            noise, 
            model_kwargs=model_kwargs, 
            uncondition_model_kwargs={'context': torch.empty((1, 0, 5)).to(noise.device)}
        )
        return y_0, y_t_hist

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        y_t, _, _, y_0_image, model_kwargs = self.process_batch(batch, mode='val')
        sampled = self.sampling(noise=torch.randn_like(y_t), model_kwargs=model_kwargs)
        return {
            'sampling': sampled
        }

class DDIM_LDM_pretrained_COCO_instance_prompt(DDIM_LDM_pretrained_COCO):
    def process_batch(self, batch, mode='train'):
        y_t, target, t, y_0, model_kwargs = super().process_batch(batch[0], mode)
        model_kwargs.update({'context': torch.tensor(batch[1])})
        return y_t, target, t, y_0, model_kwargs

class DDIM_LDM_LAION_pretrained_COCO(DDIM_LDM_pretrained_COCO):
    def training_step(self, batch, batch_idx):
        res_dict = super().training_step(batch, batch_idx)
        res_dict['model_input'] = res_dict['model_input'][:, :3] # the LAION pretrained model has 4 channels, for visualization with wandb, we only keep the first 3 channels
        return res_dict

class DDIM_LDM_LAION_pretrained_COCO_instance_prompt(DDIM_LDM_LAION_pretrained_COCO):
    def process_batch(self, batch, mode='train'):
        y_t, target, t, y_0, model_kwargs = super().process_batch(batch[0], mode)
        model_kwargs.update({'context': {
            'layout':torch.tensor(batch[1])
        }})
        return y_t, target, t, y_0, model_kwargs

class DDIM_LDM_LAION_Text(DDIM_LDM_Text_VQVAETraining):
    def initialize_unet(self, unet_init_weights):
        if unet_init_weights is not None:
            print(f'INFO: initialize denoising UNet from {unet_init_weights}, NOTE: without partial attention layers')
            model_sd = torch.load(unet_init_weights)
            self_model_sd = self.denoise_fn.state_dict()
            self_model_params = list(self.denoise_fn.named_parameters())
            self_model_k = list(map(lambda x: x[0], self_model_params))
            self.params_not_pretrained = []
            self.params_pretrained = []
            for k_idx in range(len(self_model_k)):
                this_k = self_model_k[k_idx]
                if this_k not in model_sd:
                    if this_k=='output_blocks.5.3.conv.weight':
                        self_model_sd[this_k] = model_sd['output_blocks.5.2.conv.weight']
                        self.params_pretrained.append(self_model_params[k_idx][1])
                    elif this_k=='output_blocks.5.3.conv.bias':
                        self_model_sd[this_k] = model_sd['output_blocks.5.2.conv.bias']
                        self.params_pretrained.append(self_model_params[k_idx][1])
                    elif this_k=='output_blocks.8.3.conv.weight':
                        self_model_sd[this_k] = model_sd['output_blocks.8.2.conv.weight']
                        self.params_pretrained.append(self_model_params[k_idx][1])
                    elif this_k=='output_blocks.8.3.conv.bias':
                        self_model_sd[this_k] = model_sd['output_blocks.8.2.conv.bias']
                        self.params_pretrained.append(self_model_params[k_idx][1])
                    else:
                        self.params_not_pretrained.append(self_model_params[k_idx][1])
                        continue
                elif (self_model_sd[this_k].shape == model_sd[this_k].shape) or (self_model_sd[this_k].shape == model_sd[this_k].squeeze(-1).shape):
                    self_model_sd[this_k] = model_sd[this_k]
                    self.params_pretrained.append(self_model_params[k_idx][1])
                else:
                    self.params_not_pretrained.append(self_model_params[k_idx][1])
                    continue

            self.denoise_fn.load_state_dict(self_model_sd)


    def process_batch(self, batch, mode='train'):
        y_t, target, t, y_0, model_kwargs = super().process_batch(batch[0], mode)
        model_kwargs.update({'context': {
            'layout': torch.tensor(batch[1]),
            'text': self.encode_text(batch[2])
        }})
        return y_t, target, t, y_0, model_kwargs

    def training_step(self, batch, batch_idx):
        res_dict = super().training_step(batch, batch_idx)
        res_dict['model_input'] = res_dict['model_input'][:, :3] # the LAION pretrained model has 4 channels, for visualization with wandb, we only keep the first 3 channels
        return res_dict

    @torch.no_grad()
    def fast_sampling(self, noise, model_kwargs={}):
        y_0, y_t_hist = super().fast_sampling(
            noise, 
            model_kwargs=model_kwargs, 
            uncondition_model_kwargs={'context': {
                    'layout': torch.empty((1, 0, 5)).to(noise.device),
                    'text': self.encode_text([''])
                }
            }
        )
        return y_0, y_t_hist
