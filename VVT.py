import torch
import torch.nn as nn
import warnings
import math
import torchvision.transforms as T


class VisionTransformer(nn.Module):
    #"visio-tactile transformer"
    #"depth changed from 8 to 6"
    def __init__(self, img_size=[84], patch_size=14, in_chans=3, num_classes=0, embed_dim=384, depth=6,
                 num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()

        self.num_feature = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chan=in_chans, embeded_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        self.contact_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.align_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.tacttile_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.Init_block = Init_Block(
                dim=embed_dim, num_heads=num_heads,mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,  norm_layer=norm_layer)
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads,mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm1 = norm_layer(embed_dim)
        # classifier head/change head for other tasks
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.align_token, std=.02)
        trunc_normal_(self.contact_token, std=.02)

        self.linear_img = nn.Sequential(nn.Linear(embed_dim, embed_dim//4),
                                          nn.LeakyReLU(0.2, inplace=True),
                                          nn.Linear(embed_dim//4, embed_dim//12))

        self.final_layers = nn.Sequential(nn.Linear(1152, 640),
                                          nn.LeakyReLU(0.2, inplace=True),
                                          nn.Linear(640, 288))
        self.norm2 = norm_layer(288)
        self.align_recognition = nn.Sequential(nn.Linear(embed_dim, 1),
                                               nn.Sigmoid())

        self.contact_recognition = nn.Sequential(nn.Linear(embed_dim, 1),
                                                 nn.Sigmoid())

        self.align_recognition = nn.Sequential(nn.Linear(288, 1),
                                               nn.Sigmoid())

        self.contact_recognition = nn.Sequential(nn.Linear(288, 1),
                                                 nn.Sigmoid())

        self.force_preprocess = nn.Tanh()
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w: int, h: int):
        npatch = x.shape[2] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        else:
            raise ValueError('Position Encoder does not match dimension')

    def prepare_tokens(self, x, tactile):
        B, S, nc, w, h = x.shape
        # may be later, see how original solution goes
        # if Data_Augmentation:
        #     x = x.view(B*S, nc, w, h)
        #     x = self.Data_Augmentation(x)
        #     x = x.view(B, S, nc, w, h)
        # tactile = tactile.view(B * S, -1)
        # tactile = self.force_preprocess(tactile)
        # tactile = tactile.view(B, S, -1)
        x, patched_tactile = self.patch_embed(x, tactile)
        x = x + self.pos_embed
        patched_tactile = patched_tactile + self.tacttile_pos_embed
        return self.pos_drop(x), patched_tactile

    def forward(self, x, tactile):
        x, tactile = self.prepare_tokens(x, tactile)
        x = self.Init_block(x, tactile)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm1(x)
        img_tactile = self.linear_img(x)
        B, S, patches, dim = img_tactile.size()
        img_tactile = img_tactile.view(B, S, -1)
        # print(img_tactile.size())
        img_tactile = self.final_layers(img_tactile)
        # one is to classify contact, one is to classify alignment
        return img_tactile, self.align_recognition(img_tactile), self.contact_recognition(img_tactile),

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, S, N, C = x.shape
        qkv = self.qkv(x).reshape(B*S, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, S, N, C)
        attn = attn.view(B, S, -1, N, N)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Init_Image_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, 2 * dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, tactile):
        B, S, N, C = x.shape
        q = self.q(x).reshape(B*S, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # print(self.kv(tactile).size())
        kv = self.kv(tactile).reshape(B*S, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # print(kv.size())
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, S, N, C)
        attn = attn.view(B, S, -1, N, N)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Init_Tactile_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, 2 * dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, tactile, x):
        B, S, N, C = x.shape
        q = self.q(x).reshape(B*S, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # print(self.kv(tactile).size())
        kv = self.kv(tactile).reshape(B*S, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # print(kv.size())
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, S, N, C)
        attn = attn.view(B, S, -1, N, N)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class PatchEmbed(nn.Module):
    def __init__(self, img_size=84, tactile_dim = 6, patch_size=14, in_chan=3, embeded_dim=384):
        super().__init__()
        self.num_patches = int((img_size/patch_size)*(img_size/patch_size))
        self.img_size = img_size
        self.patch_size = patch_size
        self.embeded_dim = embeded_dim
        self.proj = nn.Conv2d(in_chan, embeded_dim, kernel_size=patch_size, stride=patch_size)
        self.tactile_patch = self.num_patches
        # to prevent overfitting
        self.decode_tactile = nn.Sequential(nn.Linear(tactile_dim, self.tactile_patch*embeded_dim))

    def forward(self, image, tactile):
        # Input shape batch, Sequence, in_Channels H#W
        # Output shape batch, Sequence, correlation & out_Channels
        B, S, C, H, W = image.shape
        image = image.view(B * S, C, H, W)
        pached_image = self.proj(image).flatten(2).transpose(1, 2).view(B, S, -1, self.embeded_dim)
        tactile = tactile.view(B*S, -1)
        decoded_tactile = self.decode_tactile(tactile).view(B, S, self.tactile_patch, -1)
        # print(decoded_tactile.size())
        return pached_image, decoded_tactile


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                              proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim*mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

    def forward(self, x, return_attention: bool = False):
        y = self.attn(self.norm1(x))
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class Init_Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.img_attn = Init_Image_Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                              proj_drop=drop)
        self.tac_attn = Init_Tactile_Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                             attn_drop=attn_drop,
                                             proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim*mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.final = nn.Linear(dim*2, dim)
    def forward(self, x, tactile):
        img_atten = self.img_attn(self.norm1(x), self.norm2(tactile))
        tac_atten = self.tac_attn(self.norm1(x), self.norm2(tactile))
        x = self.final(torch.cat((img_atten, tac_atten), dim=3))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,)*(x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.MLP = nn.Sequential(nn.Linear(in_features, hidden_features),
                            act_layer(),
                            nn.Dropout(drop),
                            nn.Linear(hidden_features, out_features),
                            nn.Dropout(drop))

    def forward(self, x):
        x = self.MLP(x)
        return x


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor
