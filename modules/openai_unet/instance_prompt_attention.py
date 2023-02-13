from abc import abstractmethod
import math
import torch
import torch.nn as nn
from einops import rearrange, repeat
from .util import zero_module, checkpoint

from modules.openai_unet.partial_attention import ConditionBlock

class LearnableClassPrompt(nn.Module):
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

    @property
    def device(self):
        return next(self.parameters()).device

class SegmentationPrompt(LearnableClassPrompt):
    def forward(self, inp, shape=None):
        '''
        inp: a segmentation mask of shape b, H, W
        return: (b, H, W, d)
        '''
        inp = torch.where(inp == 255, self.num_classes, inp) # ignored region becomes null
        return self.class_embedder(inp)

class LayoutPrompt(LearnableClassPrompt):
    def forward(self, inp, shape):
        '''
        inp: a list of [x, y, h, w, obj_idx]
        return: (b, H, W, d), batch b always equals to 1
        '''
        H, W = shape
        # null embedding
        null_idx = self.num_classes
        null_idx = torch.ones(1, *shape, dtype=torch.long).to(self.device) * null_idx
        null_embedding = self.class_embedder(null_idx)

        # foreground embedding
        fg_mask = torch.zeros(1, *shape, 1, dtype=torch.long).to(self.device)
        fg_embedding = torch.zeros(1, *shape, self.embedding_dim, dtype=torch.float).to(self.device)
        inp = rearrange(inp, 'b n d -> (b n) d') # b should be 1
        for x, y, w, h, obj_idx in inp:
            i_start, i_end, j_start, j_end = self.get_region([x, y, w, h], shape=shape)
            fg_mask[:, i_start: i_end, j_start: j_end] += 1
            fg_idx = torch.ones(1, 1, 1, dtype=torch.long).to(self.device) * (obj_idx).to(torch.long)
            fg_embedding[:, i_start: i_end, j_start: j_end] += self.class_embedder(fg_idx)

        merged_embedding = torch.where(fg_mask>0.5, fg_embedding, null_embedding)
        embedding_counter = torch.where(fg_mask>0.5, fg_mask, 1)
        merged_embedding = merged_embedding / embedding_counter

        return merged_embedding

    def get_region(self, bbox, shape):
        '''
        bboxes: [[x_min, y_min, width, hight], ...]
        '''
        h, w = shape
        x_min, y_min, width, height = bbox
        i_start = min(h-1, torch.floor(y_min * h))
        j_start = min(w-1, torch.floor(x_min * w))
        i_end = i_start + max(1, torch.ceil(height * h))
        j_end = j_start + max(1, torch.ceil(width * w))
        i_start, i_end, j_start, j_end = map(int, [i_start, i_end, j_start, j_end])
        return i_start, i_end, j_start, j_end

class InstancePromptAttentionBlock(ConditionBlock):
    def __init__(
        self,
        class_embedder,
        channels,
        num_heads=1,
        num_head_channels=-1,
        num_groups=32,
        attention_intense=1.,
        instance_prompt_attn_type='segmentation',
        use_checkpoint=False,
        **kwargs
    ):
        super().__init__()

        self.checkpoint = use_checkpoint
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

        if instance_prompt_attn_type in ['layout_partial']:
            Attention=PartialLayoutQKVAttention_v2
        elif instance_prompt_attn_type in ['layout_partial_v2']: # for ablation
            Attention=PartialLayoutQKVAttention_v2
        elif instance_prompt_attn_type in ['layout_partial_v1']: # for ablation
            Attention=PartialLayoutQKVAttention_v1
        elif instance_prompt_attn_type in ['layout_no_partial']: # for ablation
            Attention=QKVAttention
        else:
            Attention=QKVAttention
        # print(f'INFO: the instance prompt type is {Attention.__name__}')

        self.attention = Attention(
            self.num_heads,
            channels,
            class_embedder,
            attention_intense, 
        )

        self.proj_out = zero_module(nn.Conv1d(channels, channels, 1))

    def forward(self, x, context):
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self, x, context):
        input_x = x.clone()
        b, c, x_h, x_w = x.shape

        qkv = self.qkv(self.norm(x).view(b, c, -1))

        h = self.attention(qkv, context, (x_h, x_w))
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
        class_embedder, # this is the instance prompt
        attention_intense=1.,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.class_embedder = class_embedder
        ch = channels // n_heads
        self.to_prompt_token = nn.Conv1d(class_embedder.embedding_dim, self.n_heads*3*ch, 1, bias=False)
        self.attention_intense = attention_intense

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, qkv, context=None, image_size=None):
        """
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :param context: [N x h x w] for segmentation, [bboxes] for layout
        :param image_size: tuple for [H, W]
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        scale = 1 / math.sqrt(math.sqrt(ch))
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        # q, k, v: bxH, C, hxw

        # get prompt
        if context is not None:
            prompt_embedding = self.class_embedder(inp=context, shape=image_size)
            prompt_embedding = rearrange(prompt_embedding, 'b h w d -> b d (h w)')
            prompt_tokens = self.to_prompt_token(prompt_embedding)
            prompt_q, prompt_k, prompt_v = prompt_tokens.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
            q = q + prompt_q
            k = k + prompt_k
            v = v + prompt_v

        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)

        return a.reshape(bs, -1, length)

class PartialLayoutQKVAttention_v2(QKVAttention):
    def forward(self, qkv, context=None, image_size=None):
        """
        Apply partial QKV attention for layout2image
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :param context: bboxes for layout, in shape [bs, num obj, 5]
        :param image_size: tuple for [H, W]
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        scale = 1 / math.sqrt(math.sqrt(ch))
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        # q, k, v: bxH, C, hxw
        
        # get null embedding
        null_context = torch.empty((bs, 0, 5), device=self.device)
        null_prompt_embedding = self.class_embedder(inp=null_context, shape=image_size)
        null_prompt_embedding = rearrange(null_prompt_embedding, 'b h w d -> b d (h w)')
        null_prompt_tokens = self.to_prompt_token(null_prompt_embedding)
        null_prompt_q, null_prompt_k, null_prompt_v = null_prompt_tokens.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        # null_prompt_q = null_prompt_k = null_prompt_v = 0 # ablation for no null token

        weight = torch.einsum(
            "bct,bcs->bts", 
            (q + null_prompt_q) * scale, 
            (k + null_prompt_k) * scale
        )
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum(
            "bts,bcs->bct", 
            weight, 
            (v + null_prompt_v)
        )

        def extract_region(tensor, region):
            return rearrange(tensor, 'c l -> l c')[region]

        # get prompt
        if context is not None:
            outputs = []

            prompt_embedding = self.class_embedder(inp=context, shape=image_size) # bs, h, w, embed dim
            prompt_embedding = rearrange(prompt_embedding, 'b h w d -> b d (h w)')
            prompt_tokens = self.to_prompt_token(prompt_embedding)
            prompt_q, prompt_k, prompt_v = prompt_tokens.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)

            # repeat context n_head times
            context = repeat(context, 'b n d -> (b head) n d', head=self.n_heads)

            for this_q, this_k, this_v, this_a_null, this_prompt_q, this_prompt_k, this_prompt_v, this_context in zip(q, k, v, a, prompt_q, prompt_k, prompt_v, context):
                this_a_null = rearrange(this_a_null, 'c l -> l c').clone() # l=length=hxw, c=ch
                this_a_fg = torch.zeros_like(this_a_null)
                this_a_fg_mask = torch.zeros((length, 1), dtype=torch.float, device=self.device)
                for x, y, h, w, obj_idx in this_context:
                    obj_loc = self.get_region([x, y, h, w], image_size=image_size)
                    this_qs = extract_region(this_q + this_prompt_q, obj_loc) #length * ch
                    this_ks = extract_region(this_k + this_prompt_k, obj_loc)
                    this_vs = extract_region(this_v + this_prompt_v, obj_loc)
                    this_weight = torch.einsum(
                        "tc,sc->ts", 
                        this_qs * scale, 
                        this_ks * scale
                    )
                    this_weight = torch.softmax(this_weight.float(), dim=-1).type(this_weight.dtype)
                    this_as = this_weight @ this_vs
                    this_a_fg[obj_loc] += this_as
                    this_a_fg_mask[obj_loc] += 1.
                this_a_final = torch.where(this_a_fg_mask > 0.5, this_a_fg, this_a_null)
                this_counter = torch.where(this_a_fg_mask > 0.5, this_a_fg_mask, 1)
                outputs.append(
                    rearrange(this_a_final / this_counter, 'l c -> c l')
                )
            a = torch.stack(outputs, dim=0)
        return a.reshape(bs, -1, length)

    def get_region(self, bbox, image_size):
        h, w = image_size
        region_mask = torch.zeros((h, w), device=self.device)
        x_min, y_min, width, height = bbox
        i_start = min(h-1, torch.floor(y_min * h))
        j_start = min(w-1, torch.floor(x_min * w))
        i_end = i_start + max(1, torch.ceil(height * h))
        j_end = j_start + max(1, torch.ceil(width * w))
        i_start, i_end, j_start, j_end = map(int, [i_start, i_end, j_start, j_end])
        region_mask[i_start: i_end, j_start: j_end] = 1.
        region_mask = region_mask.view(-1)
        obj_loc = torch.where(region_mask > 0.5)
        return obj_loc

class PartialLayoutQKVAttention_v1(QKVAttention):
    def forward(self, qkv, context=None, image_size=None):
        """
        Apply partial QKV attention for layout2image
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :param context: bboxes for layout, in shape [bs, num obj, 5]
        :param image_size: tuple for [H, W]
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        scale = 1 / math.sqrt(math.sqrt(ch))
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        # q, k, v: bxH, C, hxw
        
        # get null embedding
        null_context = torch.empty((bs, 0, 5), device=self.device)
        null_prompt_embedding = self.class_embedder(inp=null_context, shape=image_size)
        null_prompt_embedding = rearrange(null_prompt_embedding, 'b h w d -> b d (h w)')
        null_prompt_tokens = self.to_prompt_token(null_prompt_embedding)
        null_prompt_q, null_prompt_k, null_prompt_v = null_prompt_tokens.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        # null_prompt_q = null_prompt_k = null_prompt_v = 0 # ablation for no null token

        weight = torch.einsum(
            "bct,bcs->bts", 
            (q + null_prompt_q) * scale, 
            (k + null_prompt_k) * scale
        )
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum(
            "bts,bcs->bct", 
            weight, 
            (v + null_prompt_v)
        )

        def extract_region(tensor, region):
            return rearrange(tensor, 'c l -> l c')[region]

        # get prompt
        if context is not None:
            outputs = []

            prompt_embedding = self.class_embedder(inp=context, shape=image_size) # bs, h, w, embed dim
            prompt_embedding = rearrange(prompt_embedding, 'b h w d -> b d (h w)')
            prompt_tokens = self.to_prompt_token(prompt_embedding)
            prompt_q, prompt_k, prompt_v = prompt_tokens.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)

            # repeat context n_head times
            context = repeat(context, 'b n d -> (b head) n d', head=self.n_heads)

            for this_q, this_k, this_v, this_a, this_prompt_q, this_prompt_k, this_prompt_v, this_context in zip(q, k, v, a, prompt_q, prompt_k, prompt_v, context):
                this_a = this_a.permute(1, 0).clone()
                # TODO check context shape should be bs, num_obj, 5
                this_counter = torch.ones((length, ), dtype=torch.float, device=self.device)
                for x, y, h, w, obj_idx in this_context:
                    obj_loc = self.get_region([x, y, h, w], image_size=image_size)
                    this_qs = extract_region(this_q + this_prompt_q, obj_loc) #length * ch
                    this_ks = extract_region(this_k + this_prompt_k, obj_loc)
                    this_vs = extract_region(this_v + this_prompt_v, obj_loc)
                    this_weight = torch.einsum(
                        "tc,sc->ts", 
                        this_qs * scale, 
                        this_ks * scale
                    )
                    this_weight = torch.softmax(this_weight.float(), dim=-1).type(this_weight.dtype)
                    this_as = this_weight @ this_vs
                    this_a[obj_loc] += this_as
                    this_counter[obj_loc] += 1.
                outputs.append(
                    rearrange(this_a / this_counter[:, None], 'l c -> c l')
                )
            a = torch.stack(outputs, dim=0)
        return a.reshape(bs, -1, length)

    def get_region(self, bbox, image_size):
        h, w = image_size
        region_mask = torch.zeros((h, w), device=self.device)
        x_min, y_min, width, height = bbox
        i_start = min(h-1, torch.floor(y_min * h))
        j_start = min(w-1, torch.floor(x_min * w))
        i_end = i_start + max(1, torch.ceil(height * h))
        j_end = j_start + max(1, torch.ceil(width * w))
        i_start, i_end, j_start, j_end = map(int, [i_start, i_end, j_start, j_end])
        region_mask[i_start: i_end, j_start: j_end] = 1.
        region_mask = region_mask.view(-1)
        obj_loc = torch.where(region_mask > 0.5)
        return obj_loc
