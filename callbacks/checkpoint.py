from pytorch_lightning.callbacks import ModelCheckpoint, Callback
import os

def get_epoch_checkpoint(
    expt_path,
    every_n_epochs=10,
    save_top_k=5
):
    epoch_checkpoint = ModelCheckpoint(
        every_n_epochs=every_n_epochs,
        save_top_k=save_top_k,
        monitor="epoch",
        mode="max",
        dirpath=expt_path,
        filename="{epoch:04d}",
    )
    return epoch_checkpoint

def get_latest_checkpoint(
    expt_path,
    every_n_epochs=1
):
    latest_checkpoint = ModelCheckpoint(
        every_n_epochs=every_n_epochs,
        save_top_k=1,
        monitor="epoch",
        mode="max",
        dirpath=expt_path,
        filename="latest",
    )
    return latest_checkpoint

class CheckpointEveryNSteps(Callback):
    """
    from https://github.com/Lightning-AI/lightning/issues/2534
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """
    def __init__(
        self,
        expt_path,
        save_step_at,
        prefix="N-Step-Checkpoint",
        use_modelcheckpoint_filename=False,
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        self.expt_path = expt_path
        self.save_step_at = save_step_at
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename
        self.saved_steps = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """ Check if we should save a checkpoint after every train batch """
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        rank = pl_module.global_rank
        if rank == 0:
            print(global_step)
        if (global_step in self.save_step_at) and (global_step not in self.saved_steps) and (rank == 0):
            if self.use_modelcheckpoint_filename:
                filename = trainer.checkpoint_callback.filename
            else:
                filename = f"{self.prefix}_{epoch=}_{global_step=}.ckpt"
            ckpt_path = os.path.join(self.expt_path, filename)
            trainer.save_checkpoint(ckpt_path)
            self.saved_steps.append(global_step)

            if global_step == self.save_step_at[-1]:
                # training is done
                raise

def get_iteration_checkpoint(
    expt_path,
):
    print("INFO: Add iteration callbacks")
    return CheckpointEveryNSteps(
        expt_path = expt_path,
        save_step_at=[100, 200, 500, 1000, 2000]
    )