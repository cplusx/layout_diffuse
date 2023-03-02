### Custom Training
If you want to train on your dataset, you may need following knowledge
#### 1 [main.py](main.py)
The entrance of the program for training. It does following things:
* Create denoising/vqvae/text models in the config json. The denoising/vqvae/text model is a regular `pytorch module`.
* Create a DDIM training instance which is a `pytorch lightning module`. (e.g., the training class for COCO is `DDIM_LDM_LAION_Text`, you can find the class in json config file)
* Prepare dataset and dataloader.
* Create callbacks for checkpointing and visualization (see [callbacks README](callbacks/README.md) for details).
* Create a `pytorch lightning` `Trainer` instance for training. 

#### 2 [Denoising model](modules)
The denoising model is a [UNet model](modules/openai_unet/openaimodel_layout_diffuse.py) that takes layout information and (optional) text prompts.

#### 3 [Latent diffusion model](DDIM_ldm)
The folder contains the code for diffusion.
Class [DDIM_LDM](DDIM_ldm/DDIM_ldm.py) contains the coefficients and functions for diffusion and denoising process. 

Class [DDIM_LDMTraining](DDIM_ldm/DDIM_ldm.py) contains the code for
* Training (need to follow pl gramma)
* Validation/testing (need to follow pl gramma)
* Sampling
* Initializing optimizer

Class [DDIM_LDM_VQVAETraining](DDIM_ldm/DDIM_ldm.py) adds on VQVAE encoder and decoder.

Class [DDIM_LDM_Text_VQVAETraining](DDIM_ldm/DDIM_ldm.py) adds on text model

In most of the cases, you only need to overwrite the `DDIM_LDM_VQVAETraining` or `DDIM_LDM_Text_VQVAETraining` class for a customized training.

You can see class `DDIM_LDM_LAION_Text` to understand how to derive these class for each dataset/task.

Functions `trianing_step`, `validation_step` and `test_step` will return a dictionary. This dictonary will be the `outputs` arguments in the callback functions. You can use this dictionary for visualization e.t.c.


#### 4 [Callbacks](callbacks)
see callbacks' [readme](../callbacks/README.md)

#### 5 [Data](data)
The function of this folder is to return a training loader or validation loader. 

In most of the cases you can use the off-the-shelf datasets (e.g. official ones in `torchvison`). The only thing you need to modify is to overwrite the `process_batch()` funcation in `DDIM_LDM_VQVAETraining`.
