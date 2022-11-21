from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat

from .util import checkpoint


def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0., memory_efficient=False):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
        self.memory_efficient = memory_efficient

    def qkv_attention(self, q, k, v, mask=None):
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=self.heads)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        return out

    def memory_efficient_qkv_attention(self, q, k, v, mask=None):
        # note mask is not added here
        from xformers.ops import memory_efficient_attention
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        out = memory_efficient_attention(q, k, v)
        return out

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        if self.memory_efficient:
            out = self.memory_efficient_qkv_attention(q, k, v)
        else:
            out = self.qkv_attention(q, k, v, mask)

        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        return x + x_in

# class TaskPromptCrossAttention(nn.Module):
#     def __init__(self, query_dim, prompt_dim=None, num_prompt=64, heads=8, dim_head=64, dropout=0.):
#         super().__init__()
#         inner_dim = dim_head * heads
#         prompt_dim = default(prompt_dim, query_dim)

#         self.prompt = nn.Parameter(torch.randn(1, num_prompt, prompt_dim))
#         self.prompt_proj = nn.Linear(prompt_dim, query_dim, bias=False)

#         self.scale = dim_head ** -0.5
#         self.heads = heads

#         self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
#         self.to_k = nn.Linear(query_dim, inner_dim, bias=False)
#         self.to_v = nn.Linear(query_dim, inner_dim, bias=False)

#         self.to_out = nn.Sequential(
#             nn.Linear(inner_dim, query_dim),
#             nn.Dropout(dropout)
#         )


#     def qkv_attention(self, q, k, v, mask=None):
#         sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

#         if exists(mask):
#             mask = rearrange(mask, 'b ... -> b (...)')
#             max_neg_value = -torch.finfo(sim.dtype).max
#             mask = repeat(mask, 'b j -> (b h) () j', h=self.heads)
#             sim.masked_fill_(~mask, max_neg_value)

#         # attention, what we cannot get enough of
#         attn = sim.softmax(dim=-1)

#         out = einsum('b i j, b j d -> b i d', attn, v)
#         return out

#     def forward(self, x, mask=None):
#         h = self.heads

#         q = self.to_q(x)
#         prompt = self.prompt_proj(self.prompt)
#         k = self.to_k(torch.cat([prompt, x], dim=1))
#         v = self.to_v(torch.cat([prompt, x], dim=1))

#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

#         out = self.qkv_attention(q, k, v, mask)
#         out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
#         return self.to_out(out)

# class TaskPromptTransformerBlock(nn.Module):
#     def __init__(self, dim, n_heads, d_head, dropout=0., num_prompt=64, prompt_dim=None, gated_ff=True, checkpoint=True, memory_efficient=False):
#         super().__init__()
#         self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout, memory_efficient=memory_efficient)  # is a self-attention
#         self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
#         self.attn2 = TaskPromptCrossAttention(
#             query_dim=dim, 
#             prompt_dim=prompt_dim, num_prompt=num_prompt,
#             heads=n_heads, dim_head=d_head, dropout=dropout,
#             memory_efficient=memory_efficient
#         )
#         self.norm1 = nn.LayerNorm(dim)
#         self.norm2 = nn.LayerNorm(dim)
#         self.norm3 = nn.LayerNorm(dim)
#         self.checkpoint = checkpoint

#     def forward(self, x):
#         return checkpoint(self._forward, (x, ), self.parameters(), self.checkpoint)

#     def _forward(self, x):
#         x = self.attn1(self.norm1(x)) + x
#         x = self.attn2(self.norm2(x)) + x
#         x = self.ff(self.norm3(x)) + x
#         return x

# class TaskPromptSpatialTransformer(nn.Module):
#     def __init__(
#         self, 
#         in_channels, 
#         n_heads, 
#         d_head,
#         depth=1, 
#         dropout=0., 
#         num_prompt=64,
#         prompt_dim=None,
#         memory_efficient=False
#     ):
#         super().__init__()
#         self.in_channels = in_channels
#         inner_dim = n_heads * d_head
#         self.norm = Normalize(in_channels)

#         self.proj_in = nn.Conv2d(in_channels,
#                                  inner_dim,
#                                  kernel_size=1,
#                                  stride=1,
#                                  padding=0)

#         self.transformer_blocks = nn.ModuleList(
#             [TaskPromptTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, num_prompt=num_prompt, prompt_dim=prompt_dim, memory_efficient=memory_efficient)
#                 for d in range(depth)]
#         )

#         self.proj_out = zero_module(nn.Conv2d(inner_dim,
#                                               in_channels,
#                                               kernel_size=1,
#                                               stride=1,
#                                               padding=0))

#     def forward(self, x):
#         b, c, h, w = x.shape
#         x_in = x
#         x = self.norm(x)
#         x = self.proj_in(x)
#         x = rearrange(x, 'b c h w -> b (h w) c')
#         for block in self.transformer_blocks:
#             x = block(x)
#         x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
#         x = self.proj_out(x)
#         return x + x_in

class TextTaskPromptCrossAttention(nn.Module):
    def __init__(self, query_dim, prompt_dim=None, num_prompt=64, text_context_dim=None, image_in_kv=False, heads=8, dim_head=64, dropout=0.):
        '''
        if prompt_dim > 0, then will add soft task prompt
        if image_in_kv is True, then will use image in k and v
        '''
        super().__init__()
        inner_dim = dim_head * heads
        text_context_dim = default(text_context_dim, query_dim)
        self.num_prompt = num_prompt
        self.image_in_kv = image_in_kv

        if num_prompt > 0:
            prompt_dim = default(prompt_dim, query_dim)
            self.prompt = nn.Parameter(torch.randn(1, num_prompt, prompt_dim))
            self.prompt_to_context_dim = nn.Linear(prompt_dim, text_context_dim, bias=False)
            # zero_module(self.prompt_to_context_dim)

        if image_in_kv:
            self.image_to_context_dim = nn.Linear(query_dim, text_context_dim, bias=False)
            # zero_module(self.image_to_context_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(text_context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(text_context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def qkv_attention(self, q, k, v, mask=None):
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=self.heads)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        return out

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)

        if self.num_prompt > 0:
            prompt = self.prompt_to_context_dim(self.prompt)
            context = torch.cat([context, prompt], dim=1)
        
        if self.image_in_kv:
            image_token = self.image_to_context_dim(x)
            context = torch.cat([context, image_token], dim=1)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        out = self.qkv_attention(q, k, v, mask)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

class TextTaskPromptTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., num_prompt=64, prompt_dim=None, text_context_dim=None, image_in_kv=False, gated_ff=True, checkpoint=True):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout, )
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = TextTaskPromptCrossAttention(
            query_dim=dim, 
            prompt_dim=prompt_dim, num_prompt=num_prompt,
            heads=n_heads, dim_head=d_head, dropout=dropout,
            text_context_dim=text_context_dim, image_in_kv=image_in_kv
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context):
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self, x, context):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x

class TextTaskPromptSpatialTransformer(nn.Module):
    def __init__(
        self, 
        in_channels, 
        n_heads, 
        d_head,
        depth=1, 
        dropout=0., 
        num_prompt=64,
        prompt_dim=None,
        text_context_dim=None,
        image_in_kv=False
    ):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [TextTaskPromptTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, num_prompt=num_prompt, prompt_dim=prompt_dim, text_context_dim=text_context_dim,
            image_in_kv=image_in_kv)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

    def forward(self, x, context=None):
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        return x + x_in


