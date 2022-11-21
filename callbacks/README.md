### [Checkpoint callbacks](checkpoint.py)
* get_epoch_checkpoint: save checkpoint every `n` epochs, name after `epoch={:04d}.ckpt`
* get_latest_checkpoint: save the lastest checkpoint, name after `latest.ckpt`

### [WandB callbacks](wandb.py)
Visualize the input and output images
The `outputs` argument is a dictionary that contains return from training model
See [DDPM.py](../DDPM/DDPM.py) for the items in the dictionary