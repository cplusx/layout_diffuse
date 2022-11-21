from abc import abstractmethod
import math
import torch
import torch.nn as nn

from .util import zero_module

class ConditionBlock(nn.Module):
    @abstractmethod
    def forward(self, x, z):
        """
        Apply the module to `x` given `z` condition.
        The condition can be a random noise or trainable latent such as VAE
        """

class LearnableClassEmbedding(nn.Module):
    def __init__(
        self, 
        num_classes, 
        embedding_dim=128,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.class_embedder = nn.Embedding(
            num_embeddings=num_classes+1,
            embedding_dim=embedding_dim
        )
        self.embedding_dim = embedding_dim

    def forward(self, idx, shape):
        '''
        idx: long
        shape: target shape
        '''
        if idx is None:
            idx = self.num_classes # the last embedding is the null token
        idx = torch.ones(*shape, dtype=torch.long).to(self.device()) * idx
        return self.class_embedder(idx)

    def device(self):
        return next(self.parameters()).device


class PartialAttentionBlock(ConditionBlock):
    def __init__(
        self,
        class_embedder,
        channels,
        num_heads=1,
        num_head_channels=-1,
        num_groups=32,
        attention_type='segmentation',
        attention_intense=1.,
        **kwargs
    ):
        super().__init__()

        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.norm = nn.GroupNorm(num_groups, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)

        if attention_type in ['segmentation']:
            Attention = SegmentationQKVAttention
        elif attention_type in ['layout']:
            Attention = LayoutQKVAttention
        elif attention_type in ['layout_batch']:
            Attention = LayoutBatchQKVAttention
        elif attention_type in ['perceiver']:
            Attention = PerceiverQKVAttention
        else:
            raise NotImplementedError

        self.attention = Attention(
            self.num_heads,
            channels,
            class_embedder,
            attention_intense, 
        )

        # self.proj_out = zero_module(nn.Conv1d(channels, channels, 1))
        self.proj_out = nn.Conv1d(channels, channels, 1)

    def forward(self, x, attn_masks=None):
        b, c, x_h, x_w = x.shape

        qkv = self.qkv(self.norm(x).view(b, c, -1))

        h = self.attention(qkv, attn_masks, (x_h, x_w))
        h = self.proj_out(h)
        return x + h.reshape(b, c, x_h, x_w)

class QKVAttention(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(
        self, 
        n_heads, 
        channels,
        class_embedder,
        attention_intense=1.,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.class_embedder = class_embedder
        ch = channels // n_heads
        self.to_cls_token = nn.Conv1d(class_embedder.embedding_dim, 3*ch, 1, bias=False)
        self.attention_intense = attention_intense

    def device(self):
        return next(self.parameters()).device

    def format_and_duplicate_mask_n_head_times(self, attn_mask):
        raise NotImplementedError

    def get_region(self, idx, attn_mask):
        raise NotImplementedError

    def get_correct_embedding(self, obj_idx, this_attn_mask, shape):
        raise NotImplementedError

    def forward(self, qkv, attn_masks=None, image_size=None):
        """
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :param attn_mask: [N x h x w] for segmentation, {idx: bboxes} for layout
        :param cls_token: [N x (3 * C) x T]
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        scale = 1 / math.sqrt(math.sqrt(ch))
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        # q, k, v: bxH, C, hxw

        '''basic qkv attention with null token'''
        cls_embedding = self.get_correct_embedding(
            None, 
            attn_masks,
            shape=(bs * self.n_heads, length)
        )
        cls_token = self.to_cls_token(cls_embedding.permute(0, 2, 1)) # b*H, 3xch, hxw
        q_null, k_null, v_null = cls_token.split(ch, dim=1)
        weight = torch.einsum(
            "bct,bcs->bts", (q+q_null) * scale, (k+k_null) * scale
        )
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, (v+v_null))

        if attn_masks is None:
            return a.reshape(bs, -1, length)

        '''partial qkv attention with object class token'''
        attn_masks = self.format_and_duplicate_mask_n_head_times(attn_masks)
        assert len(q) == len(k) == len(v) == len(a) == len(attn_masks), f'got {len(q)} and {len(attn_masks)}'

        # for loop for each sample and each object in each batch
        # has to do in a for-loop since each object has different num of pixels thus cannot be processed in a batch
        outs = [] # this is the output
        for this_qs, this_ks, this_vs, this_as, this_attn_mask in zip(q, k, v, a, attn_masks):
            # this_qs, this_ks, this_vs, this_as: of shape ch, length
            # this_attn_mask: of shape length
            this_qs = this_qs.permute(1, 0) # length, ch
            this_ks = this_ks.permute(1, 0) # length, ch
            this_vs = this_vs.permute(1, 0) # length, ch
            this_as = this_as.permute(1, 0).clone() # length, ch
            this_counter = torch.ones(this_qs.shape[0], dtype=torch.float, device=this_qs.device)

            index_list = self.get_index_list(this_attn_mask)
            for obj_idx in index_list:
                obj_loc, obj_shape = self.get_region(obj_idx, this_attn_mask, image_size)
                this_q = this_qs[obj_loc]; this_k = this_ks[obj_loc]; this_v = this_vs[obj_loc]

                cls_embedding = self.get_correct_embedding(
                    obj_idx, 
                    this_attn_mask, 
                    shape=obj_shape
                )
                cls_token = self.to_cls_token(cls_embedding.permute(1, 0)[None])[0] # 3xch, length
                q_cls, k_cls, v_cls = cls_token.permute(1, 0).split(ch, dim=1) # length, ch

                weight = torch.einsum(
                    "tc,sc->ts", 
                    (this_q + q_cls) * scale, 
                    (this_k + k_cls) * scale
                )
                weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
                this_a = weight @ (this_v + v_cls) # num_this_objxnum_this_obj @ num_this_objxch -> num_this_objxch
                this_as[obj_loc] += this_a
                this_counter[obj_loc] += self.attention_intense
            outs.append((this_as / this_counter[:, None]).permute(1, 0)) # ch, length
        a = torch.stack(outs, dim=1) # is this wrong??? should be dim=0???
        return a.reshape(bs, -1, length)


class SegmentationQKVAttention(QKVAttention):
    def format_and_duplicate_mask_n_head_times(self, attn_masks):
        bs, h, w = attn_masks.shape
        attn_masks = attn_masks.view(-1, h*w)
        attn_masks = torch.stack([attn_masks]*self.n_heads, dim=1).view(bs*self.n_heads, h*w)
        return attn_masks

    def get_index_list(self, this_attn_mask):
        index_list = torch.unique(this_attn_mask)
        return index_list

    def get_region(self, idx, attn_mask, image_size=None):
        # return loc, shape
        obj_loc = torch.where(attn_mask == idx)
        return obj_loc, attn_mask[obj_loc].shape

    def get_correct_embedding(self, obj_idx, this_attn_mask, shape):
        cls_embedding = self.class_embedder(obj_idx, shape=shape) # length[obj_loc], ch
        return cls_embedding

class LayoutQKVAttention(QKVAttention):
    def format_and_duplicate_mask_n_head_times(self, bbox_dict):
        #  note: in layout mode, the batch size will always be 1
        bbox_list = [bbox_dict] * self.n_heads
        return bbox_list

    def get_index_list(self, this_attn_mask):
        index_list = torch.tensor(list(this_attn_mask.keys()), device=self.device())
        return index_list

    def get_region(self, idx, bbox_dicts, image_size):
        '''
        bboxes: [[x_min, y_min, width, hight], ...]
        '''
        bboxes = bbox_dicts[int(idx)]
        h, w = image_size
        region_mask = torch.zeros((h, w), device=idx.device)
        for x_min, y_min, width, height in bboxes:
            i_start = min(h-1, torch.floor(y_min * h))
            j_start = min(w-1, torch.floor(x_min * w))
            i_end = i_start + max(1, torch.ceil(height * h))
            j_end = j_start + max(1, torch.ceil(width * w))
            i_start, i_end, j_start, j_end = map(int, [i_start, i_end, j_start, j_end])
            region_mask[i_start: i_end, j_start: j_end] = 1.
        region_mask = region_mask.view(-1)
        obj_loc = torch.where(region_mask > 0.5)
        return obj_loc, region_mask[obj_loc].shape

    def get_correct_embedding(self, obj_idx, this_attn_mask, shape):
        cls_embedding = self.class_embedder(obj_idx, shape=shape) # length[obj_loc], ch
        return cls_embedding

class LayoutBatchQKVAttention(QKVAttention):
    def format_and_duplicate_mask_n_head_times(self, bbox_dict):
        bbox_list = [bbox_dict] * self.n_heads
        return bbox_list

    def get_index_list(self, this_attn_mask):
        index_list = torch.tensor(list(this_attn_mask.keys()), device=self.device())
        return index_list

    def get_region(self, idx, bbox_dicts, image_size):
        '''
        bboxes: [[x_min, y_min, width, hight], ...]
        '''
        bboxes = bbox_dicts[int(idx)]
        h, w = image_size
        region_mask = torch.zeros((h, w), device=idx.device)
        for x_min, y_min, width, height in bboxes:
            i_start = min(h-1, torch.floor(y_min * h))
            j_start = min(w-1, torch.floor(x_min * w))
            i_end = i_start + max(1, torch.ceil(height * h))
            j_end = j_start + max(1, torch.ceil(width * w))
            i_start, i_end, j_start, j_end = map(int, [i_start, i_end, j_start, j_end])
            region_mask[i_start: i_end, j_start: j_end] = 1.
        region_mask = region_mask.view(-1)
        obj_loc = torch.where(region_mask > 0.5)
        return obj_loc, region_mask[obj_loc].shape

    def get_correct_embedding(self, obj_idx, this_attn_mask, shape):
        cls_embedding = self.class_embedder(obj_idx, shape=shape) # length[obj_loc], ch
        return cls_embedding
        
class PerceiverQKVAttention(QKVAttention):
    def __init__(self, n_heads, channels, class_embedder, attention_intense=1):
        super().__init__(n_heads, channels, class_embedder, attention_intense)

    def format_and_duplicate_mask_n_head_times(self, sub_images_and_bboxes):
        # convert sub_images to sub_image_perceiver_features, note the batch size can only be one
        # sub_images: b, num_objs, 3, h, w
        # bboxes: b, num_objs, 4
        sub_images, bboxes = sub_images_and_bboxes
        if len(sub_images) == 0: # when there is no object in the image
            return [[sub_images, bboxes]] * self.n_heads
        b, num_objs, dim, h, w = sub_images.shape
        sub_images = sub_images.view(b*num_objs, dim, h, w)
        cls_embedding = self.class_embedder(sub_images).view(b, num_objs, -1)
        assert bboxes.shape[1] == num_objs, f'got {bboxes.shape} and {num_objs}'
        res = []
        for sample_idx in range(b):
            for head_idx in range(self.n_heads):
                res.append([cls_embedding[sample_idx], bboxes[sample_idx]])
        return res

    def get_index_list(self, this_attn_mask):
        index_list = torch.arange(len(this_attn_mask[0]), device=self.device()) # num of objects
        return index_list

    def get_region(self, idx, sub_images_and_bboxes, image_size=None):
        # return loc, shape
        _, bboxes = sub_images_and_bboxes
        bbox = bboxes[int(idx)]
        h, w = image_size
        region_mask = torch.zeros((h, w), device=self.device())
        x_min, y_min, width, height = bbox
        i_start = min(h-1, torch.floor(y_min * h))
        j_start = min(w-1, torch.floor(x_min * w))
        i_end = i_start + max(1, torch.ceil(height * h))
        j_end = j_start + max(1, torch.ceil(width * w))
        i_start, i_end, j_start, j_end = map(int, [i_start, i_end, j_start, j_end])
        region_mask[i_start: i_end, j_start: j_end] = 1.
        region_mask = region_mask.view(-1)
        obj_loc = torch.where(region_mask > 0.5)
        return obj_loc, region_mask[obj_loc].shape

    def get_correct_embedding(self, obj_idx, this_attn_mask, shape):
        '''
        whole image case:
            obj_idx: None
            return: an all zero feature
        sub image case:
            obj_idx: an int
            this_attn_mask: [(num_obj, feat_dim), (num_obj, 4)]
            return: a feature of shape (*shape, feat_dim)
        '''
        if obj_idx is None:
            return torch.zeros(*shape, self.class_embedder.embedding_dim, device=self.device())
        obj_embedding, bbox = this_attn_mask # (feat_dim, ), (4, )
        obj_embedding, bbox = obj_embedding[obj_idx], bbox[obj_idx]
        feat_dim = obj_embedding.shape[-1]
        return obj_embedding.view(*[1]*len(shape), feat_dim) # (1, 1, ..., feat_dim)