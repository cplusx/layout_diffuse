import os
import torch
import torchvision
import cv2
import math
import numpy as np
from pytorch_lightning.callbacks import Callback
from ..sampling_save_fig import save_figure, save_sampling_history
from data.coco_detect import get_coco_id_mapping
from data.vg import get_vg_id_mapping

def format_image(x):
    x = x.cpu()
    x = (x + 1) / 2
    x = x.clamp(0, 1)
    x = x.permute(1,2,0).detach().numpy()
    return x

def save_raw_image_tensor(x, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path, image=x)

def generate_choice(im_h, im_w, p1, p2, w, h):
    up_outside = p1[1] - h >= 3
    down_outside = p2[1] + h <= im_h - 3
    left_outside = p2[0] - w >= 3
    p_choice = []
    if up_outside:
        cp1 = p1[0], p1[1] - h - 3
        cp2 = p1[0] + w, p1[1]
        cpt = p1[0], p1[1] - 2
        p_choice.append([cp1, cp2, cpt])
        if left_outside:
            cp1 = p2[0] - w, p1[1] - h - 3
            cp2 = p2[0], p1[1]
            cpt = p2[0] - w, p1[1] - 2
            p_choice.append([cp1, cp2, cpt])
    cp1 = p1[0], p1[1]
    cp2 = p1[0] + w, p1[1] + h + 3
    cpt = p1[0], p1[1] + h + 2
    p_choice.append([cp1, cp2, cpt])
    if left_outside:
        cp1 = p2[0] - w, p1[1]
        cp2 = p2[0], p1[1] + h + 3
        cpt = p2[0] - w, p1[1] + h + 2
        p_choice.append([cp1, cp2, cpt])
    cp1 = p1[0], p2[1] - h - 3
    cp2 = p1[0] + w, p2[1]
    cpt = p1[0], p2[1] - 2
    p_choice.append([cp1, cp2, cpt])
    if left_outside:
        cp1 = p2[0] - w, p2[1] - h - 3
        cp2 = p2[0], p2[1]
        cpt = p2[0] - w, p2[1] - 2
        p_choice.append([cp1, cp2, cpt])
    if down_outside:
        cp1 = p1[0], p2[1]
        cp2 = p1[0] + w, p2[1] + h + 3
        cpt = p1[0], p2[1] + h + 2
        p_choice.append([cp1, cp2, cpt])
        if left_outside:
            cp1 = p2[0] - w, p2[1]
            cp2 = p2[0], p2[1] + h + 3
            cpt = p2[0] - w , p2[1] + h + 2
            p_choice.append([cp1, cp2, cpt])
    return p_choice

def check_choice(choice, state):
    p1, p2, pt = choice[:3]
    for per_state in state:
        sp1, sp2, spt = per_state[:3]
        if p2[0] <= sp1[0] or p2[1] <= sp1[1] or p1[0] >= sp2[0] or p1[1] >= sp2[1]:
            continue
        return False
    return True

def search_text_position(choices, state):
    for choice in choices[len(state)]:
        if check_choice(choice, state):
            if len(state) == len(choices) - 1:
                return state + (choice, )
            else:
                result = search_text_position(choices, state + (choice, ))
                if result is not None:
                    return result
                else:
                    continue
    return None

def plot_bbox_without_overlap(image, bboxes, color_mapper):
    H, W = image.shape[:2]
    lw = max(round(sum(image.shape) / 2 * 0.003), 2)

    choices, colors, class_names = [], [], []
    for bbox in bboxes:
        x, y, w, h = bbox[:4]
        x, y, w, h = list(map(int, [x*W, y*H, w*W, h*H]))
        label = int(bbox[-1]) if len(bbox) == 5 else None
        if label <= 0:
            continue
        color, class_name = color_mapper(label)
        p1, p2 = (int(x), int(y)), (int(x + w), int(y + h))
        image = cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
        if label:
            tf = max(lw - 1, 1)  
            textw, texth = cv2.getTextSize(class_name, 0, fontScale=lw / 3, thickness=tf)[0] 
            p_choice = generate_choice(H, W, p1, p2, textw, texth)
            choices.append(p_choice)
            colors.append(color)
            class_names.append(class_name)
    
    position_result = search_text_position(choices, state=())
    # print(position_result)
    if position_result is None:
        return None

    for position, color, class_name in zip(position_result, colors, class_names):
        p1, p2, pt = position
        cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  
        cv2.putText(
            image,
            class_name, 
            pt,
            0,
            lw / 3,
            (1., 1., 1.),
            thickness=tf,
            lineType=cv2.LINE_AA
        )
    return image
def plot_bounding_box(image, bboxes, color_mapper):
    # bboxes: num_obj, 5
    H, W = image.shape[:2]
    for bbox in bboxes:
        x, y, w, h = bbox[:4]
        x, y, w, h = list(map(int, [x*W, y*H, w*W, h*H]))
        label = int(bbox[-1]) + 1 if len(bbox) == 5 else None
        # in the network, we let label start from 0 by -1, now we add 1 back
        color, class_name = color_mapper(label)
        # plot the rectangle bounding box and label
        image = cv2.rectangle(image, (int(x), int(y)), (int(x+w), int(y+h)), color, 2)
        if label:
            (w, h), _ = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)
            image = cv2.rectangle(image, (x, y+20), (x + w, y), color, -1)
            cv2.putText(image, class_name, (x, y+18), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (1,1,1), 1)
    return image

def save_bounding_box(bboxes, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as OUT:
        for i, bbox in enumerate(bboxes):
            x, y, w, h, obj_idx = bbox
            x, y, w, h = list(map(float, [x, y, w, h]))
            obj_idx = int(obj_idx) # this obj_idx starts from 0 (i.e. all index are coco idx - 1)
            OUT.write(f'{x},{y},{w},{h},{obj_idx}')
            if i < len(bboxes) - 1:
                OUT.write('\n')

class ColorMapping():
    def __init__(self, id_class_mapping, mesh_dim=3):
        self.id_class_mapping = id_class_mapping
        num_classes = len(id_class_mapping)
        num_grid_each_dim = math.ceil(num_classes**(1/mesh_dim))
        mesh_d = np.meshgrid(
            *[np.linspace(0,1,num_grid_each_dim)]*mesh_dim
        )
        mesh_d = [i.reshape(-1) for i in mesh_d]
        self.mesh = np.stack(mesh_d, axis=-1)

        self.id_to_mesh_idx = {}
        for idx, (class_id, class_name) in enumerate(id_class_mapping.items()):
            self.id_to_mesh_idx[class_id] = idx
    
    def __call__(self, class_id):
        class_name = self.id_class_mapping[class_id]
        mesh_index = self.id_to_mesh_idx[class_id]
        return self.mesh[mesh_index], class_name

class LayoutImageSavingCallback(Callback):
    def __init__(self):
        raise NotImplementedError

    def overlap_image_with_bbox(self, image, bbox):
        image_with_bbox = plot_bbox_without_overlap(
            image.copy(),
            bbox,
            self.label_color_mapper
        ) if len(bbox) <= 10 else None
        if image_with_bbox is not None:
            return image_with_bbox
        return plot_bounding_box(
            image.copy(), 
            bbox,
            self.label_color_mapper
        )

    def save_y_0_hat(
        self, gt_image, sample_image, 
        bbox, rank, 
        current_epoch, current_idx, 
        num_gpus=1, num_nodes=1, appendix=''
    ):
        save_name = f'{rank+num_nodes*num_gpus*current_idx:04d}_{self.repeat_idx:02d}'
        save_bounding_box(
            bbox,
            save_path=os.path.join(
                self.expt_path, 
                f'epoch_{current_epoch:05d}' + appendix,
                'bounding_box', 
                f'{save_name}.txt')
        )
        # save gt + bbox
        save_figure(
            self.overlap_image_with_bbox(
                format_image(gt_image),
                bbox
            ),
            save_path=os.path.join(
                self.expt_path, 
                f'epoch_{current_epoch:05d}' + appendix,
                'gt_image', 
                f'{save_name}.jpg')
        )
        # save sampling
        save_figure(
            format_image(sample_image),
            save_path=os.path.join(
                self.expt_path, 
                f'epoch_{current_epoch:05d}' + appendix,
                'sample_image', 
                f'{save_name}.jpg')
        )
        # save sampling + bbox
        save_figure(
            self.overlap_image_with_bbox(
                format_image(sample_image),
                bbox
            ),
            save_path=os.path.join(
                self.expt_path, 
                f'epoch_{current_epoch:05d}' + appendix,
                'sample_image_with_bbox', 
                f'{save_name}.jpg')
        )
        # save blank canvas + bbox
        save_figure(
            self.overlap_image_with_bbox(
                np.ones_like(format_image(sample_image)),
                bbox
            ),
            save_path=os.path.join(
                self.expt_path, 
                f'epoch_{current_epoch:05d}' + appendix,
                'layout', 
                f'{save_name}.jpg')
        )
        # save raw image tensor
        save_raw_image_tensor(
            format_image(sample_image),
            save_path=os.path.join(
                self.expt_path, 
                f'epoch_{current_epoch:05d}' + appendix,
                'raw_tensor', 
                f'{save_name}') # will add .npz automatically
        )

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if batch_idx == 0:
            self.repeat_idx += 1
            self.current_idx = 0
        rank = pl_module.global_rank
        current_epoch = pl_module.current_epoch

        y_0_hat = outputs['sampling']['model_output']
        y_t_hist = outputs['sampling']['model_history_output']

        gt_images = batch[0]
        bboxes = batch[1]

        if pl_module.use_fast_sampling:
            sampler = pl_module.fast_sampler
            sampling_steps = pl_module.fast_sampling_steps
            guidance_sacle = pl_module.guidance_scale
            appendix = f'_{sampler}_{sampling_steps}_{guidance_sacle:.1f}'
        else:
            appendix=''

        for gt_image, image, bbox in zip(gt_images, y_0_hat, bboxes):
            self.save_y_0_hat(
                gt_image, image, bbox,
                rank=rank, current_epoch=current_epoch,
                current_idx = self.current_idx,
                num_gpus=trainer.num_devices,
                num_nodes=trainer.num_nodes,
                appendix=appendix
            )
        # self.save_y_t_hist(
        #     y_t_hist,
        #     prefix='sampling_hist',
        #     rank=rank, current_epoch=current_epoch,
        #     current_idx = self.current_idx
        # )

            self.current_idx += 1

class COCOLayoutImageSavingCallback(LayoutImageSavingCallback):
    def __init__(self, expt_path, start_idx=0):
        self.expt_path = expt_path
        self.current_idx = start_idx
        coco_id_mapping = get_coco_id_mapping()
        print(coco_id_mapping)
        self.label_color_mapper = ColorMapping(id_class_mapping=coco_id_mapping)
        self.repeat_idx = -1

class VGLayoutImageSavingCallback(LayoutImageSavingCallback):
    def __init__(self, expt_path, start_idx=0):
        self.expt_path = expt_path
        self.current_idx = start_idx
        vg_id_mapping = get_vg_id_mapping()
        print(vg_id_mapping)
        self.label_color_mapper = ColorMapping(id_class_mapping=vg_id_mapping)
        self.repeat_idx = -1