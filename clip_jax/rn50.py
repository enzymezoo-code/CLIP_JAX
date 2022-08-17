import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import math
import os
import clip
import hashlib
import torch

# bn has an is_training parameter

def _download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError("Model has been downloaded but the SHA256 checksum does not not match")

    return download_target

def create_bn(name):
    init = hk.initializers.RandomUniform()
    return hk.BatchNorm(True, True, 0.1, scale_init=init, offset_init=init, name=name)

class PositionalEmbedding(hk.Module):
    def __init__(self, spacial_dim, embed_dim):
        super().__init__(name="pos_e")
        self.positional_embedding = hk.get_parameter("pos_embd", (spacial_dim ** 2 + 1, embed_dim), init=hk.initializers.RandomNormal())
    def __call__(self, x):
        return x + self.positional_embedding[None, :, :]

class AttentionPool2d(hk.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__(name="attnpool")
        self.pe = PositionalEmbedding(spacial_dim, embed_dim)
        self.num_heads = num_heads
        self.mha = hk.MultiHeadAttention(num_heads, 
        embed_dim // num_heads, 
        w_init_scale=0.1, 
        value_size=None, 
        model_size=output_dim or embed_dim, 
        name="mha")

    def __call__(self, x):
        N, H, W, C = x.shape
        # NHWC -> (HW) NC
        x = x.reshape(N, H * W, C).transpose(0, 1, 2)
        #x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        n = x.mean(axis=1, keepdims=True)
        x = jax.numpy.concatenate([n, x], axis=1)  # (HW+1)NC
        x = self.pe(x) 
        x = self.mha(x[:1], x, x)
        return x

class Bottleneck(hk.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, layer_num=0, num=0):
        super().__init__(name=f'layer_{layer_num}_{num}')

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = hk.Conv2D(planes, kernel_shape=(1, 1), stride=1, padding="SAME", with_bias=False, name="conv1")
        self.bn1 = create_bn(name="bn1")
        self.conv2 = hk.Conv2D(planes, kernel_shape=(3, 3), stride=1, padding="SAME", with_bias=False, name="conv2")
        self.bn2 = create_bn(name="bn2")
        self.avgpool = hk.AvgPool(stride, stride, padding="SAME") if stride > 1 else None
        self.conv3 = hk.Conv2D(planes * self.expansion, kernel_shape=(1, 1), stride=1, padding="SAME", with_bias=False, name="conv3")
        self.bn3 = create_bn(name="bn3")
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = hk.Sequential([
                hk.AvgPool(stride, stride, padding="SAME"),
                hk.Conv2D(planes * self.expansion, kernel_shape=(1, 1), stride=1, with_bias=False, name="downsample_conv")
            ]
            )
            self.downsample_bn = hk.BatchNorm(True, True, 0.1, name="downsample_bn")
        

    def __call__(self, x):
        identity = x
        out = jax.nn.relu(self.bn1(self.conv1(x), True))
        out = jax.nn.relu(self.bn2(self.conv2(out), True))
        if self.stride > 1:
            out = self.avgpool(out)
        out = self.bn3(self.conv3(out), True)

        if self.downsample is not None:
            identity = self.downsample(x)
            identity = self.downsample_bn(identity, True)
        
        out += identity
        out = jax.nn.relu(out)
        return out


@hk.transparent
class ModifiedResNet(hk.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64, name=None):
        super().__init__(name="visual")
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = hk.Conv2D(width // 2, 
            kernel_shape=(3, 3), 
            stride=(2, 2), 
            padding="SAME", 
            with_bias=False, 
            name="conv1")
        self.bn1 = create_bn(name="bn1")

        self.conv2 = hk.Conv2D(width // 2, 
            kernel_shape=(3, 3), 
            stride=(1, 1), 
            padding="SAME", 
            with_bias=False,
            name="conv2")
        self.bn2 = create_bn(name="bn2")

        self.conv3 = hk.Conv2D(width,
            kernel_shape=(3, 3), 
            stride=(1, 1), 
            padding="SAME", 
            with_bias=False,
            name="conv3")
        self.bn3 = create_bn(name="bn3")

        self.avgpool = hk.AvgPool(2, 2, padding="SAME")

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0], layer_num=1)
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2, layer_num=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2, layer_num=3)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2, layer_num=4)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1, layer_num=0):
        layers = [Bottleneck(self._inplanes, planes, stride, layer_num=layer_num, num=0)]

        self._inplanes = planes * Bottleneck.expansion
        for i in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes, layer_num=layer_num, num=i))

        return hk.Sequential(layers)

    def __call__(self, x):
        def stem(x):
            x = jax.nn.relu(self.bn1(self.conv1(x), True))
            x = jax.nn.relu(self.bn2(self.conv2(x), True))
            x = jax.nn.relu(self.bn3(self.conv3(x), True))
            x = self.avgpool(x)
            return x

        #x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)
        return x

def forward_function(x):
    layers = [3, 4, 6, 3]
    output_size = 1024
    heads = 8
    model = ModifiedResNet(layers, output_size, heads)
    return model(x)
        
def load_weights_rn50(state_dict, pytree):

    atom_map_r = {
        'weight': 'w',
        'bias': 'b' ,
    }

    atom_map = {}
    atom_map['w'] = 'weight'
    atom_map['b'] = 'bias'
    atom_map['scale'] = 'weight'
    atom_map['offset'] = 'bias'
    atom_map['query'] = 'q_proj'
    atom_map['key'] = 'k_proj'
    atom_map['value'] = 'v_proj'
    atom_map['linear'] = 'c_proj'

    # load stem weights
    #print(pytree.keys())
    stem_path = "/visual"

    for key in pytree.keys():
        # stem weights
        if 'make_layer' not in key and 'attnpool' not in key:
            layer_name = key.split('/')[-1]
            state_dict_key_base = 'visual.' + layer_name 
            for sub_layer, value in pytree[key].items():
                ind = state_dict_key_base + '.' + atom_map[sub_layer]
                if len(state_dict[ind].shape) == 4:
                    new_val = jnp.array(state_dict[ind], dtype=jnp.float32).transpose()
                else:
                    new_val = jnp.array(state_dict[ind], dtype=jnp.float32).reshape(value.shape)
                pytree[key][sub_layer] = new_val
        # bottleneck weights            
        if '_make_layer' in key and 'downsample' not in key:
            print(key)
            layer_name = key.split('/')[-1]
            bottleneck_names = key.split('/')[-3]
            b_name_1 = int(bottleneck_names[-3])
            b_name_2 = int(bottleneck_names[-1])

            state_dict_key_base = f'visual.layer{b_name_1}.{b_name_2}.{layer_name}'

            for sub_layer, value in pytree[key].items():
                ind = state_dict_key_base + '.' + atom_map[sub_layer]
                if len(state_dict[ind].shape) == 4:
                    new_val = jnp.array(state_dict[ind], dtype=jnp.float32).transpose()
                else:
                    new_val = jnp.array(state_dict[ind], dtype=jnp.float32).reshape(value.shape)
        # downsample bottleneck weights
        if 'downsample' in key:
            layer_name = key.split('/')[-1]
            bottleneck_names = key.split('/')[-3]
            b_name_1 = int(bottleneck_names[-3])
            b_name_2 = int(bottleneck_names[-1])
            state_dict_key_base = f'visual.layer{b_name_1}.{b_name_2}.downsample.'
            if 'downsample_conv' in key:
                state_dict_key_base  += '0.'
            elif 'downsample_bn' in key:
                state_dict_key_base  += '1.'
            
            for sub_layer, value in pytree[key].items():
                ind = state_dict_key_base + atom_map[sub_layer] 
                if len(state_dict[ind].shape) == 4:
                    new_val = jnp.array(state_dict[ind], dtype=jnp.float32).transpose()
                else:
                    new_val = jnp.array(state_dict[ind], dtype=jnp.float32).reshape(value.shape)
                pytree[key][sub_layer] = new_val
        
        # attnpool weights
        if 'attnpool' in key:
            if 'pos_e' in key:
                ind = 'visual.attnpool.positional_embedding'
                new_val = jnp.array(state_dict[ind], dtype=jnp.float32)
                pytree[key]['pos_embd'] = new_val
            else:
                attn_layer_name = key.split('/')[-1]
                state_dict_key_base = f'visual.attnpool.{atom_map[attn_layer_name]}.'
                for sub_layer, value in pytree[key].items():
                    ind = state_dict_key_base + atom_map[sub_layer] 
                    if len(state_dict[ind].shape) == 4:
                        new_val = jnp.array(state_dict[ind], dtype=jnp.float32).transpose()
                    else:
                        new_val = jnp.array(state_dict[ind], dtype=jnp.float32).reshape(value.shape)
                    pytree[key][sub_layer] = new_val

def count_params(pytree: dict, root="/"):
    total = 0
    for k, v in pytree.items():
        if isinstance(v, dict):
            total += count_params(pytree[k], root + "/" + k + "/")
        else:
            print(f'{root + k} {v.shape}')
            total += math.prod(v.shape)
    return total

if __name__ == "__main__":
    layers = [3, 4, 6, 3]
    output_size = 7
    heads = 8
    key = hk.PRNGSequence(42)
    forward = hk.transform_with_state(forward_function)
    x = jnp.ones((1, 224, 224, 3))
    params, state = forward.init(next(key), x)

    loc = _download("https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt", os.path.expanduser("~/.cache/clip"))
    with open(loc, "rb") as f:
        state_dict = torch.jit.load(loc, map_location="cpu").eval().state_dict()
    load_weights_rn50(state_dict, params)
    #count_params(params)
    y = forward.apply(params, state, None, x)
