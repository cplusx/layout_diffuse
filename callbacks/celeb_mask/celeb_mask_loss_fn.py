'''Change the loss weight of background as the training continues
At the beginning of each epoch, update loss function
'''
from pytorch_lightning.callbacks import Callback
from functools import partial

class CelebBackgroundLossWeightTuner(Callback):
    def __init__(self, max_epochs, loss_fn, num_classes):
        super().__init__()
        self.max_epochs = max_epochs
        self.loss_fn = loss_fn
        self.num_classes = num_classes

    def on_epoch_start(self, trainer, pl_module):
        current_epoch = pl_module.current_epoch
        this_bg_weight = current_epoch / self.max_epochs
        print(f'INFO: set background weight to {this_bg_weight}')
        this_loss_fn = partial(
            self.loss_fn, 
            background_weight=this_bg_weight,
            num_classes=self.num_classes
        )
        pl_module.loss_fn = this_loss_fn
        return super().on_epoch_start(trainer, pl_module)