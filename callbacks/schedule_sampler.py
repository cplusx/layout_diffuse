from pytorch_lightning.callbacks import Callback
from DDIM.schedule_sampler import LossSecondMomentResampler

'''May use in the future, now directly change schedule sampler in the DDIM.py'''
class ScheduleSamplerCallback(Callback):
    '''change the sample weight of different timestamps'''
    def on_train_batch_end(
        self, trainer, pl_module, 
        outputs, batch, batch_idx
    ):
        if isinstance(pl_module.schedule_sampler, LossSecondMomentResampler):
            t = outputs['t']
            loss_flat = outputs['loss_flat']
            pl_module.schedule_sampler.update_with_all_losses(
                t, loss_flat.detach()
            )