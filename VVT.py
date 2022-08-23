import torch
import torch.nn as nn
import warnings
import math
import torchvision.transforms as T


class VisionTransformer(nn.Module):
    " adjust depth?"
    def __init__(self, img_size=[84], patch_size=14, in_chans=3, num_classes=0, embed_dim=384, image_depth=4, tactile_depth=4, visual_tactile_depth=4,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.num_feature = self.embed_dim = embed_dim

        self.vision_patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chan=in_chans, embeded_dim=embed_dim
        )
        self.tactile_patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chan=in_chans, embeded_dim=embed_dim
        )

        vision_num_patches = self.vision_patch_embed.num_patches
        tactile_num_patches = self.tactile_patch_embed.tactile_patch

        self.v_vision_pos_embed = nn.Parameter(torch.zeros(1, vision_num_patches, embed_dim))
        self.v_vision_pos_drop = nn.Dropout(p=drop_rate)
        self.v_tactile_pos_embed = nn.Parameter(torch.zeros(1, tactile_num_patches, embed_dim))
        self.v_tactile_pos_drop = nn.Dropout(p=drop_rate)

        self.t_vision_pos_embed = nn.Parameter(torch.zeros(1, vision_num_patches, embed_dim))
        self.t_vision_pos_drop = nn.Dropout(p=drop_rate)
        self.t_tactile_pos_embed = nn.Parameter(torch.zeros(1, tactile_num_patches, embed_dim))
        self.t_tactile_pos_drop = nn.Dropout(p=drop_rate)

        self.contact_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.align_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.VT_pos_embed = nn.Parameter(torch.zeros(1, vision_num_patches + tactile_num_patches + 2, embed_dim))
        self.VT_pos_drop = nn.Dropout(p=drop_rate)

        vision_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, image_depth)]
        tactile_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, tactile_depth)]
        VT_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, visual_tactile_depth)]

        self.vision_blocks = nn.ModuleList([
            Vision_Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=vision_dpr[i], norm_layer=norm_layer)
            for i in range(image_depth)])

        self.tactile_blocks = nn.ModuleList([
            Tactile_Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=tactile_dpr[i], norm_layer=norm_layer)
            for i in range(tactile_depth)])

        self.VVT_blocks = nn.ModuleList([
            VVT_Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=VT_dpr[i], norm_layer=norm_layer)
            for i in range(visual_tactile_depth)])

        self.v_norm = norm_layer(embed_dim)
        self.t_norm = norm_layer(embed_dim)
        self.vt_norm = norm_layer(embed_dim)

        # classifier head/change head for other tasks
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity
        trunc_normal_(self.v_vision_pos_embed, std=.02)
        trunc_normal_(self.v_tactile_pos_embed, std=.02)
        trunc_normal_(self.t_vision_pos_embed, std=.02)
        trunc_normal_(self.t_tactile_pos_embed, std=.02)
        trunc_normal_(self.VT_pos_embed, std=0.02)

        self.linear_img = nn.Sequential(nn.Linear(embed_dim, embed_dim//4),
                                          nn.ReLU(),
                                          nn.Linear(embed_dim//4, embed_dim//12))

        self.final_layers = nn.Sequential(nn.Linear(32 * (tactile_num_patches + vision_num_patches + 2), 640),
                                          nn.ReLU(),
                                          nn.Linear(640, 288))
        self.norm2 = norm_layer(288)

        self.contact_recognition = nn.Sequential(nn.Linear(embed_dim, 1),
                                                 nn.Sigmoid())

        self.align_recognition = nn.Sequential(nn.Linear(embed_dim, 1),
                                               nn.Sigmoid())

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self):
        return self.v_vision_pos_embed, self.v_tactile_pos_embed, self.t_vision_pos_embed, self.t_tactile_pos_embed, self.VT_pos_embed

    def prepare_tokens(self, x, tactile, first_layer_trans):
        if first_layer_trans:
            v_patched_img, v_patched_tactile = self.vision_patch_embed(x, tactile)
            t_patched_img, t_patched_tactile = self.tactile_patch_embed(x, tactile)
            v_x, v_t = v_patched_img + self.interpolate_pos_encoding()[0], v_patched_tactile + self.interpolate_pos_encoding()[1]
            t_x, t_t = t_patched_img + self.interpolate_pos_encoding()[2], v_patched_tactile + self.interpolate_pos_encoding()[3]
            return self.v_vision_pos_drop(v_x), self.v_tactile_pos_drop(v_t), self.t_vision_pos_drop(t_x), self.t_tactile_pos_drop(t_t)
        else:
            B, S,_,_ = x.shape
            x = torch.cat((x, tactile), dim=2)
            alignment_tokens = self.align_token.expand(B, S, -1, -1)
            contact_tokens = self.contact_token.expand(B, S, -1, -1)
            x = torch.cat((alignment_tokens, x), dim=2)
            x = torch.cat((contact_tokens, x), dim=2)
            x = x + self.interpolate_pos_encoding()[4]
            return self.VT_pos_drop(x)

    def forward(self, x, tactile):
        v_x, v_t, t_x, t_t= self.prepare_tokens(x, tactile, True)

        for blk in self.vision_blocks:
            vision_x = blk(v_x, v_t)
        for blk in self.tactile_blocks:
            tactile_x = blk(t_x, t_t)

        vision_x = self.v_norm(vision_x)
        tactile_x = self.t_norm(tactile_x)
        x = self.prepare_tokens(vision_x, tactile_x, False)
        x = self.vt_norm(x)
        img_tactile = self.linear_img(x)
        B, S, patches, dim = img_tactile.size()
        img_tactile = img_tactile.view(B, S, -1)
        img_tactile = self.final_layers(img_tactile)
        img_tactile = self.norm2(img_tactile)
        return img_tactile, self.contact_recognition(x[:, :, 0]), self.align_recognition(x[:, :, 1])

class Image_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, 2*dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, tactile):
        B, S, N, C = x.shape
        t_B, t_S, t_N, t_C = tactile.shape
        q = self.q(x).reshape(B*S, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q = q[0]
        kv = self.kv(tactile).reshape(t_B*t_S, t_N, 2, self.num_heads, t_C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, S, N, C)
        attn = attn.view(B, S, -1, N, t_N)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

class Tactile_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, 2*dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, tactile):
        B, S, N, C = x.shape
        t_B, t_S, t_N, t_C = tactile.shape
        q = self.q(tactile).reshape(t_B*t_S, t_N, 1, self.num_heads, t_C // self.num_heads).permute(2, 0, 3, 1, 4)
        q = q[0]
        kv = self.kv(x).reshape(B*S, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(t_B, t_S, t_N, t_C)
        attn = attn.view(t_B, t_S, -1, t_N, N)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

class VT_Attention(nn.Module):
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
        return x, attn

class PatchEmbed(nn.Module):
    def __init__(self, img_size=84, tactile_dim = 6, patch_size=14, in_chan=3, embeded_dim=384):
        super().__init__()
        self.num_patches = int((img_size/patch_size)*(img_size/patch_size))
        self.img_size = img_size
        self.patch_size = patch_size
        self.embeded_dim = embeded_dim
        self.proj = nn.Conv2d(in_chan, embeded_dim, kernel_size=patch_size, stride=patch_size)
        self.tactile_patch = 3
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
        return pached_image, decoded_tactile


class Vision_Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Image_Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                              proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim*mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, tactile, return_attention: bool = False):
        y, attn = self.attn(self.norm1(x), self.norm1(tactile))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class Tactile_Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Tactile_Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                              proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim*mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, tactile, return_attention: bool = False):
        y, attn = self.attn(self.norm1(x), self.norm1(tactile))
        if return_attention:
            return attn
        tactile = tactile + self.drop_path(y)
        tactile = tactile + self.drop_path(self.mlp(self.norm2(tactile)))
        return tactile

class VVT_Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = VT_Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                              proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim*mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention: bool = False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn

        x = x + self.drop_path(y)
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


# dimension tests
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# a = torch.ones(32, 10, 3, 84, 84).to(device)
# tactile = torch.ones(32, 10, 6).to(device)
# # #
# # # # start = time.time()
# VIT = Vision_Tactile_Transformer().to(device)
# # # # # load_data_time = time.time()
# test = VIT(a, tactile)
# print(test[0].size())
# print(test.size())
# # computational_time = time.time()
# # print(computational_time - load_data_time)
# # print(load_data_time - start)
# # print(test[0].size())
# # print(test[1].size())

# Attention = Tactile_Attention(dim=384).to(device)
# patch = PatchEmbed().to(device)
# V, T = patch(a, tactile)
# attention_test = Attention(V, T)
# print(attention_test[0].size())
# # print(test.size())
# # Trans_block = Block(dim=384, num_heads=8)
# # VIT_test = Trans_block(test)
#
# # print(VIT_test[0].size())
# # print(VIT_test[1].size())
