# Model obtained from: https://github.com/NVIDIA/torch-harmonics/blob/main/torch_harmonics/examples/models/sfno.py
# Do not used, since it is a global model, which is not suitable for the regional forecast.
import os
import sys
import torch
import numpy as np
import torch.nn as nn
from torch_harmonics import *
from ._sfno_layers import *
from functools import partial
from torch import Tensor
from typing import Any, Optional

class SphericalFourierNeuralOperatorBlock(nn.Module):
    """
    Helper module for a single SFNO/FNO block. Can use both FFTs and SHTs to represent either FNO or SFNO blocks.
    """

    def __init__(
        self,
        forward_transform,
        inverse_transform,
        input_dim,
        output_dim,
        operator_type="driscoll-healy",
        mlp_ratio=2.0,
        drop_rate=0.0,
        drop_path=0.0,
        act_layer=nn.ReLU,
        norm_layer=nn.Identity,
        factorization=None,
        separable=False,
        rank=128,
        inner_skip="linear",
        outer_skip=None,
        use_mlp=True,
    ):
        super().__init__()

        if act_layer == nn.Identity:
            gain_factor = 1.0
        else:
            gain_factor = 2.0

        if inner_skip == "linear" or inner_skip == "identity":
            gain_factor /= 2.0

        self.global_conv = SpectralConvS2(forward_transform, inverse_transform, input_dim, output_dim, gain=gain_factor, operator_type=operator_type, bias=False)

        if inner_skip == "linear":
            self.inner_skip = nn.Conv2d(input_dim, output_dim, 1, 1)
            nn.init.normal_(self.inner_skip.weight, std=math.sqrt(gain_factor / input_dim))
        elif inner_skip == "identity":
            assert input_dim == output_dim
            self.inner_skip = nn.Identity()
        elif inner_skip == "none":
            pass
        else:
            raise ValueError(f"Unknown skip connection type {inner_skip}")

        # first normalisation layer
        self.norm0 = norm_layer()

        # dropout
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        gain_factor = 1.0
        if outer_skip == "linear" or inner_skip == "identity":
            gain_factor /= 2.0

        if use_mlp == True:
            mlp_hidden_dim = int(output_dim * mlp_ratio)
            self.mlp = MLP(
                in_features=output_dim, out_features=input_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop_rate=drop_rate, checkpointing=False, gain=gain_factor
            )

        if outer_skip == "linear":
            self.outer_skip = nn.Conv2d(input_dim, input_dim, 1, 1)
            torch.nn.init.normal_(self.outer_skip.weight, std=math.sqrt(gain_factor / input_dim))
        elif outer_skip == "identity":
            assert input_dim == output_dim
            self.outer_skip = nn.Identity()
        elif outer_skip == "none":
            pass
        else:
            raise ValueError(f"Unknown skip connection type {outer_skip}")

        # second normalisation layer
        self.norm1 = norm_layer()

    def forward(self, x):

        x, residual = self.global_conv(x)

        x = self.norm0(x)

        if hasattr(self, "inner_skip"):
            x = x + self.inner_skip(residual)

        if hasattr(self, "mlp"):
            x = self.mlp(x)

        x = self.norm1(x)

        x = self.drop_path(x)

        if hasattr(self, "outer_skip"):
            x = x + self.outer_skip(residual)

        return x


class SphericalFourierNeuralOperatorNet(nn.Module):
    """
    SphericalFourierNeuralOperator module. Implements the 'linear' variant of the Spherical Fourier Neural Operator
    as presented in [1]. Spherical convolutions are applied via spectral transforms to apply a geometrically consistent
    and approximately equivariant architecture.

    Parameters
    ----------
    img_shape : tuple, optional
        Shape of the input channels, by default (128, 256)
    operator_type : str, optional
        Type of operator to use ('driscoll-healy', 'diagonal'), by default "driscoll-healy"
    scale_factor : int, optional
        Scale factor to use, by default 3
    in_chans : int, optional
        Number of input channels, by default 3
    out_chans : int, optional
        Number of output channels, by default 3
    embed_dim : int, optional
        Dimension of the embeddings, by default 256
    num_layers : int, optional
        Number of layers in the network, by default 4
    activation_function : str, optional
        Activation function to use, by default "gelu"
    encoder_layers : int, optional
        Number of layers in the encoder, by default 1
    use_mlp : int, optional
        Whether to use MLPs in the SFNO blocks, by default True
    mlp_ratio : int, optional
        Ratio of MLP to use, by default 2.0
    drop_rate : float, optional
        Dropout rate, by default 0.0
    drop_path_rate : float, optional
        Dropout path rate, by default 0.0
    normalization_layer : str, optional
        Type of normalization layer to use ("layer_norm", "instance_norm", "none"), by default "instance_norm"
    hard_thresholding_fraction : float, optional
        Fraction of hard thresholding (frequency cutoff) to apply, by default 1.0
    big_skip : bool, optional
        Whether to add a single large skip connection, by default True
    pos_embed : bool, optional
        Whether to use positional embedding, by default True

    Example:
    --------
    >>> model = SphericalFourierNeuralOperatorNet(
    ...         img_size=(128, 256),
    ...         scale_factor=4,
    ...         in_chans=2,
    ...         out_chans=2,
    ...         embed_dim=16,
    ...         num_layers=4,
    ...         use_mlp=True,)
    >>> model(torch.randn(1, 2, 128, 256)).shape
    torch.Size([1, 2, 128, 256])

    References
    -----------
    .. [1] Bonev B., Kurth T., Hundt C., Pathak, J., Baust M., Kashinath K., Anandkumar A.;
        "Spherical Fourier Neural Operators: Learning Stable Dynamics on the Sphere" (2023).
        ICML 2023, https://arxiv.org/abs/2306.03838.
    """

    def __init__(
        self,
        img_size=(200, 400),
        operator_type="driscoll-healy",
        grid="equiangular",
        grid_internal="legendre-gauss",
        scale_factor=4,
        in_chans=36,
        out_chans=6,
        embed_dim=384,
        num_layers=8,
        activation_function="relu",
        encoder_layers=1,
        use_mlp=True,
        mlp_ratio=2.0,
        drop_rate=0.0,
        drop_path_rate=0.0,
        normalization_layer="none",
        hard_thresholding_fraction=1.0,
        use_complex_kernels=True,
        big_skip=True,
        pos_embed=True,
        traced_vars: Optional[list] = None,
    ):

        super().__init__()

        self.operator_type = operator_type
        self.img_size = img_size
        self.grid = grid
        self.grid_internal = grid_internal
        self.scale_factor = scale_factor
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.normalization_layer = normalization_layer
        self.use_mlp = use_mlp
        self.encoder_layers = encoder_layers
        self.big_skip = big_skip
        self.traced_vars = traced_vars

        # activation function
        if activation_function == "relu":
            self.activation_function = nn.ReLU
        elif activation_function == "gelu":
            self.activation_function = nn.GELU
        # for debugging purposes
        elif activation_function == "identity":
            self.activation_function = nn.Identity
        else:
            raise ValueError(f"Unknown activation function {activation_function}")

        # compute downsampled image size. We assume that the latitude-grid includes both poles
        self.h = (self.img_size[0] - 1) // scale_factor + 1
        self.w = self.img_size[1] // scale_factor

        # dropout
        self.pos_drop = nn.Dropout(p=drop_rate) if drop_rate > 0.0 else nn.Identity()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.num_layers)]

        # pick norm layer
        if self.normalization_layer == "layer_norm":
            norm_layer0 = partial(nn.LayerNorm, normalized_shape=(self.img_size[0], self.img_size[1]), eps=1e-6)
            norm_layer1 = partial(nn.LayerNorm, normalized_shape=(self.h, self.w), eps=1e-6)
        elif self.normalization_layer == "instance_norm":
            norm_layer0 = partial(nn.InstanceNorm2d, num_features=self.embed_dim, eps=1e-6, affine=True, track_running_stats=False)
            norm_layer1 = partial(nn.InstanceNorm2d, num_features=self.embed_dim, eps=1e-6, affine=True, track_running_stats=False)
        elif self.normalization_layer == "none":
            norm_layer0 = nn.Identity
            norm_layer1 = norm_layer0
        else:
            raise NotImplementedError(f"Error, normalization {self.normalization_layer} not implemented.")

        if pos_embed == "latlon" or pos_embed == True:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.embed_dim, self.img_size[0], self.img_size[1]))
            nn.init.constant_(self.pos_embed, 0.0)
        elif pos_embed == "lat":
            self.pos_embed = nn.Parameter(torch.zeros(1, self.embed_dim, self.img_size[0], 1))
            nn.init.constant_(self.pos_embed, 0.0)
        elif pos_embed == "const":
            self.pos_embed = nn.Parameter(torch.zeros(1, self.embed_dim, 1, 1))
            nn.init.constant_(self.pos_embed, 0.0)
        else:
            self.pos_embed = None

        # construct an encoder with num_encoder_layers
        num_encoder_layers = 1
        encoder_hidden_dim = int(self.embed_dim * mlp_ratio)
        current_dim = self.in_chans
        encoder_layers = []
        for l in range(num_encoder_layers - 1):
            fc = nn.Conv2d(current_dim, encoder_hidden_dim, 1, bias=True)
            # initialize the weights correctly
            scale = math.sqrt(2.0 / current_dim)
            nn.init.normal_(fc.weight, mean=0.0, std=scale)
            if fc.bias is not None:
                nn.init.constant_(fc.bias, 0.0)
            encoder_layers.append(fc)
            encoder_layers.append(self.activation_function())
            current_dim = encoder_hidden_dim
        fc = nn.Conv2d(current_dim, self.embed_dim, 1, bias=False)
        scale = math.sqrt(1.0 / current_dim)
        nn.init.normal_(fc.weight, mean=0.0, std=scale)
        if fc.bias is not None:
            nn.init.constant_(fc.bias, 0.0)
        encoder_layers.append(fc)
        self.encoder = nn.Sequential(*encoder_layers)

        # compute the modes for the sht
        modes_lat = self.h
        # due to some spectral artifacts with cufft, we substract one mode here
        modes_lon = (self.w // 2 + 1) -1

        modes_lat = modes_lon = int(min(modes_lat, modes_lon) * self.hard_thresholding_fraction)

        self.trans_down = RealSHT(*self.img_size, lmax=modes_lat, mmax=modes_lon, grid=self.grid).float()
        self.itrans_up = InverseRealSHT(*self.img_size, lmax=modes_lat, mmax=modes_lon, grid=self.grid).float()
        self.trans = RealSHT(self.h, self.w, lmax=modes_lat, mmax=modes_lon, grid=grid_internal).float()
        self.itrans = InverseRealSHT(self.h, self.w, lmax=modes_lat, mmax=modes_lon, grid=grid_internal).float()

        self.blocks = nn.ModuleList([])
        for i in range(self.num_layers):

            first_layer = i == 0
            last_layer = i == self.num_layers - 1

            forward_transform = self.trans_down if first_layer else self.trans
            inverse_transform = self.itrans_up if last_layer else self.itrans

            inner_skip = "none"
            outer_skip = "identity"

            if first_layer:
                norm_layer = norm_layer1
            elif last_layer:
                norm_layer = norm_layer0
            else:
                norm_layer = norm_layer1

            block = SphericalFourierNeuralOperatorBlock(
                forward_transform,
                inverse_transform,
                self.embed_dim,
                self.embed_dim,
                operator_type=self.operator_type,
                mlp_ratio=mlp_ratio,
                drop_rate=drop_rate,
                drop_path=dpr[i],
                act_layer=self.activation_function,
                norm_layer=norm_layer,
                inner_skip=inner_skip,
                outer_skip=outer_skip,
                use_mlp=use_mlp,
            )

            self.blocks.append(block)

        # construct an decoder with num_decoder_layers
        num_decoder_layers = 1
        decoder_hidden_dim = int(self.embed_dim * mlp_ratio)
        current_dim = self.embed_dim + self.big_skip * self.in_chans
        decoder_layers = []
        for l in range(num_decoder_layers - 1):
            fc = nn.Conv2d(current_dim, decoder_hidden_dim, 1, bias=True)
            # initialize the weights correctly
            scale = math.sqrt(2.0 / current_dim)
            nn.init.normal_(fc.weight, mean=0.0, std=scale)
            if fc.bias is not None:
                nn.init.constant_(fc.bias, 0.0)
            decoder_layers.append(fc)
            decoder_layers.append(self.activation_function())
            current_dim = decoder_hidden_dim
        fc = nn.Conv2d(current_dim, self.out_chans, 1, bias=False)
        scale = math.sqrt(1.0 / current_dim)
        nn.init.normal_(fc.weight, mean=0.0, std=scale)
        if fc.bias is not None:
            nn.init.constant_(fc.bias, 0.0)
        decoder_layers.append(fc)
        self.decoder = nn.Sequential(*decoder_layers)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def forward_features(self, x):
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        return x

    def forward(self,
                x: Tensor, 
    ) -> Tensor:
        
        if self.big_skip:
            residual = x
        x = self.encoder(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.forward_features(x)
        if self.big_skip:
            x = torch.cat((x, residual), dim=1)
        x = self.decoder(x)
        return x


if __name__ == "__main__":
    import os
    import torch
    import pynvml
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from omegaconf import OmegaConf
    cfg = OmegaConf.load('../correction/sfno_config.yaml')
    model_args = cfg.model_args
    data_args = cfg.data_args
    
    def print_gpu_memory():
        allocated = torch.cuda.memory_allocated(device) / 1024**2  # MB
        reserved = torch.cuda.memory_reserved(device) / 1024**2    # MB
        max_allocated = torch.cuda.max_memory_allocated(device) / 1024**2
        print(f"[Memory] Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB | Peak: {max_allocated:.2f} MB")

    # input_size = len(model_args['input_vars'] + model_args['input_cons'])
    input_size = len(model_args['input_vars']['pressure_levels'] + model_args['input_vars']['single_level'] + model_args['input_cons'])
    output_size = len(model_args['output_vars'])

    model = SphericalFourierNeuralOperatorNet(
        img_size = data_args['crop_size'],
        scale_factor = model_args['scale_factor'],
        in_chans = input_size,
        out_chans = output_size,
        embed_dim = model_args['embed_dim'],
        num_layers = model_args['num_layers'],
        use_mlp = model_args['use_mlp']
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print("After model init:")
    print_gpu_memory()

    torch.cuda.empty_cache()
    rand_input = torch.rand((1, input_size, data_args['crop_size'][0], data_args['crop_size'][1])).to(device)
    output = model(rand_input)
    print("After model forward: ")
    print_gpu_memory()
    print("Output shape: ", output.shape)