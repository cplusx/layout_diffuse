### [Checkpoint callbacks](checkpoint.py)
* get_epoch_checkpoint: save checkpoint every `n` epochs, name after `epoch={:04d}.ckpt`
* get_latest_checkpoint: save the lastest checkpoint, name after `latest.ckpt`

### Image saving callbacks
This includes `sampling_save_fig.py`, `coco_layoutsampling_save_fig.py` and `celeb_mask/sampling_save_fig.py`. These callbacks are used to save images during sampling (the output of `validation_step()` will be passed to these callbacks.)

### [WandB callbacks](wandb.py)
Visualize the input and output images
The `outputs` argument is a dictionary that contains return from `train_step()`