from .DDIM_ldm import DDIM_LDMTraining
class DDIM_LDM_CIFAR(DDIM_LDMTraining):
    def process_batch(self, y_0, mode='train'):
        return super().process_batch(y_0[0], mode)