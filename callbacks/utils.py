import torch
from data.face_parsing import MaskMeshConverter, celebAMask_labels

def unnorm(x):
    '''convert from range [-1, 1] to [0, 1]'''
    return (x+1) / 2

def clip_image(x, min=0., max=1.):
    return torch.clamp(x, min=min, max=max)

def colorize_mask(x, input_is_float=True):
    assert len(x.shape) == 4 or len(x.shape) == 5, f'mask should have shape 4 or 5, got {x.shape}'
    # input: b, 1, h, w or b, time, 1, h, w
    if len(x.shape) >= 4:
        x = x.squeeze(-3) # 
    if input_is_float:
        x = (256*x).cpu().to(torch.long) # why * 255 is not correct???
    mask_cvt = MaskMeshConverter(
        labels = list(celebAMask_labels.keys()),
        mesh_dim=3
    )
    x = mask_cvt(x) # b, h, w, 3 or b, time, h, w, 3
    if len(x.shape) == 4:
        return x.permute(0, 3, 1, 2) # b, 3, h, w
    elif len(x.shape) == 5:
        return x.permute(0, 1, 4, 2, 3) # b, time, 3, h, w
    else:
        raise RuntimeError(f'Unknown dim, mask shape is {x.shape}')

def to_mask_if_dim_gt_3(x, dim=1, keepdim=True, colorize=True):
    # TODO, now function also accept 3D mask (b, h, w), give it a new name
    '''
    valid x shape: (b, h, w), (b, 1, h, w), (b, 3, h, w), (b, num class, h, w)
    (b, t, 1, h, w), (b, t, 3, h, w), (b, t, num class, h, w)
    colorize_mask valid input shapes are (b, 1, h, w) and (b, t, 1, h, w)
    '''
    if len(x.shape) == 3:
        '''handle (b, h, w)'''
        x = x.unsqueeze(1)
    if x.shape[dim] > 3:
        '''handle (b, num class, h, w) and (b, t, num class, h, w)'''
        x = torch.argmax(x, dim=dim, keepdim=keepdim)
    if colorize:
        if x.shape[-3] != 1:
            print(f'WARNING: mask has shape {x.shape}, will not apply colorization since the dim[-3] != 1')
        else:
            x = colorize_mask(x, input_is_float=False)
    return x
