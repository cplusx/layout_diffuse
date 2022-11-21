### TL;DR
This is an implementation of diffusion model (DDPM) and improved diffusion model (DDIM) with [pytorch lightning](https://www.pytorchlightning.ai/). To start a training, the code entrance is [main.py](main.py) and the config file should be like the ones in folder [configs](configs). 
* To run a toy example of DDPM on COCO, run `python main.py -c configs/archived_configs/coco_ldm.json`. 

The visualization depends on `wandb`, remember to set it on your server by `wandb login`.

### 0. Installation
Download COCO dataset. Run `bash scripts/download_coco.sh`. This should create a folder in `~/disk2/data/COCO` and put all files COCO needs in that folder.
Next, install a conda environment that has pytorch. Follow the official instruction from the [website](https://pytorch.org/get-started/locally/). Install the correct pytorch according to your hardware.
Then git clone this folder and use pip to install requirements
```
pip install -r requirements.txt
```
You may need to modify the pytorch verison to match the CUDA driver version on your server.
Download pretrained models, run
```
bash scripts/download_pretrained_models.sh
```

To start training, use `python main.py -c [path to config file]` (e.g. `python main.py -c configs/laion_cocostuff_text_v2.json`)
NOTE: the code is integrated with wandb for visualization, if you have not set your wandb locally, it will pop up a message to ask you decide whether to record or not. 
Following configs are often modified (see the config json file):
*  "expt_name": "laion_ldm_cocostuff_layout_caption_v2", the results will be saved in the `experiment/[expt_name]`
* "devices": [0,1,2,3], gpu ids, NO need to modify if you have $CUDA_VISIBLE_DEVICES=4,5,6,7
* "data", go to folder `data/__init__.py` to find more details

### =========== below are old README, some content are obsoleted =========

### 1. Toy example of MNIST
Different datasets have different data format and preprocessing steps. To adapt the training to a specific dataset, you need to modify following things.
* The denoising model only takes in one batched images as input and outputs one batched images (i.e. only one tensor input/output). You do **NOT** need to modify the existing dataest (e.g. official dataset provided in torchvision). Instead, you overwrite the `batch_process` function in class `DDPMTraining`. Here is an [example](DDPM/DDPM_MNIST.py) for MNIST. By default, the official MNIST dataset from torchvision returns both images and labels, in the `process_batch` function, we remove the label and pass images only.
```
class DDPM_MNIST(DDPMTraining):
    def process_batch(self, y_0, mode='train'):
        return super().process_batch(y_0[0], mode)
```
* You neet to determine the training configurations in the json file. See [this](configs/MNIST.json) for an example. Specifically, you need to be aware of following arguments
    * `in_channel` and `out_channel`. They are data dependent and your training strategy (e.g. concat images) may also effect the number of input/output channels
    * `res_block_type` and `attn_block_type`. They can change the ResBlocks and AttentionBlocks used in the denosing model. If you want to modify time embedding or add additional latent, you may need to write your own blocks (put them in [modules](modules)) and pass corresponding names here.

### 2. Structure of the code

#### 2.1 [main.py](main.py)
The entrance of the program for training. It does following things:
* Create a denoising model with the `args['denoising_model']` in the config json. The denoising model is a regular `pytorch module`.
* Create a DDPMTraining instance with denoising model. The DDPMTraining is a `pytorch lightning module`.
* Prepare dataset and dataloader.
* Create callbacks for checkpointing and visualization (see [callbacks README](callbacks/README.md) for details).
* Create a `pytorch lightning` `Trainer` instance for training. 

#### 2.2 [Denoising model](modules)
The denoising model is a [UNet model](modules/unet.py) which uses [ResBlocks](modules/res_blocks.py) and [AttentionBlocks](modules/attentions.py). 

There are two [abstract classes](modules/__init__.py) need to be aware, `EmbedBlock` and `ConditionBlock`. The former one takes `time embedding` (or equivalent sqrt(alpha_prod)) while the latter one takes a latent `z` as an extra input. The latter one is the code entrance for future use and is not used for now.

To customize a `ResBlock` or `AttentionBlock`, just inherit the basic corresponding blocks and the abovementioned abstract classes, and then modifies the `forward` function. Finally, modify the blocks in the [UNet model](modules/unet.py).

#### 2.3 [DDPM](DDPM)
The folder contains the code for diffusion.
Class [DDPMBase](DDPM/DDPM.py) contains the coefficients and functions for diffusion and denoising process. 

Class [DDPMTraining](DDPM/DDPM.py) contains the code for
* Training (need to follow pl gramma)
* Validation/testing (need to follow pl gramma)
* Sampling
* Initializing optimizer

In most of the cases, you only need to overwrite the `DDPMTraining` class for a customized training. (See MNIST toy example.)

Functions `trianing_step`, `validation_step` and `test_step` will return a dictionary. This dictonary will be the `outputs` arguments in the callback functions. You can use this dictionary for visualization e.t.c.

#### 2.4 [DDIM](DDIM)
The folder contains the code for improved diffusion model. The structure is similar to DDPM. However, since DDIM's prediction target, the way of computing posterior (due to variance is predicted from model) is different than DDPM, so we keep them in two separate folders.
Class [DDIMBase](DDIM/DDIM.py) contains the coefficients and functions for diffusion and denoising process. 

Class [DDIMTraining](DDIM/DDIM.py) contains the code for
* All DDPM functions that mentioned in above section
* Loss schedule_resampler. This is proposed in DDIM to stabalize the training when L_{vlb} is used. Its function is to give different timesteps different change to be sampled during training according to the loss magnitude.
* DDIM's way to compute posterior. When computing y_t -> y_{t-1}, the variance can be (this is the default setting) computed from the network. 


#### 2.4 [Callbacks](callbacks)
see callbacks' [readme](callbacks/README.md)

#### 2.5 [Data](data)
The function of this folder is to return a training loader or validation loader. (Maybe testing loader in the future, in the scope of sampling, testing is often not the case.)

In most of the cases you can use the off-the-shelf datasets (e.g. official ones in `torchvison`). The only thing you need to modify is to overwrite the `DDPMTraining` as stated in Sec. 2.3

### 3. Exsiting issues
* There is checkpoint resume issue because of PyTorch version 1.12.0. Now only read weights for model and ignore status of optimizer (corresponding code in the [main.py](main.py)). The bug is supposed to be fixed in 1.12.1. see this [link](https://github.com/pytorch/pytorch/issues/80809#issuecomment-1175211598). Possible [solution](https://github.com/pytorch/pytorch/commit/11cadb117f65205c09f5576b433a87fcb38705df)
 by remove these lines in your torch code directly * WandB logger has issue with save directory (argument "save_dir"). By passing `None`, the wandb will create a dir named `wandb` in the root dir, this will not cause visualization error. However, if pass a `string` to it, the visualization figures will not be uploaded correctly to wandb cloud and cannot be visualized. 

### License
For open source projects, say how it is licensed.