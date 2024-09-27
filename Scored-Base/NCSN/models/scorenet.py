import torch
import functools
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet

###############################################################################
# Helper Functions
###############################################################################
def get_norm_layer(norm_type='instance'):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

##############################################################################
# Classes
##############################################################################
class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ELU()]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out
    
class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        # downrelu = nn.LeakyReLU(0.2, True)
        downrelu = nn.ELU()
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ELU()
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:  # add skip connections
            return torch.cat([x, self.model(x)], 1)

# Residual block for convolution
class ConvResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, resize=False, act='relu'):
        super().__init__()
        self.resize = resize
        
        # Choose the activation function based on the argument passed 
        def get_act():
            if act == 'relu':
                return nn.ReLU(inplace=True)
            elif act == 'softplus':
                return nn.Softplus()
            elif act == 'elu':
                return nn.ELU()
            elif act == 'leakyrelu':
                return nn.LeakyReLU(0.2, inplace=True)
        
        # If resize is not True, perform regular convolution
        # If resize is True, apply downsampling
        if not resize:
            self.main = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1),
                nn.GroupNorm(8, out_channel),
                get_act(),
                nn.Conv2d(out_channel, out_channel, 3, stride=1, padding=1),
                nn.GroupNorm(8, out_channel)
            )
        else:
            self.main = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 3, stride=2, padding=1),
                nn.GroupNorm(8, out_channel),
                get_act(),
                nn.Conv2d(out_channel, out_channel, 3, stride=1, padding=1),
                nn.GroupNorm(8, out_channel)
            )
            
        self.residual = nn.Conv2d(in_channel, out_channel, 3, stride=2, padding=1)
        self.final_act = get_act()
        
    def forward(self, inputs):
        if not self.resize:
            h = self.main(inputs)
            h += inputs
        else:
            h = self.main(inputs)
            res = self.residual(inputs)  # Residual downsample
            h += res
        return self.final_act(h)

# Deconvolutional block with residual connection
class DeconvResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, resize=False, act='relu'):
        super().__init__()
        self.resize = resize
        
        def get_act():
            if act == 'relu':
                return nn.ReLU(inplace=True)
            elif act == 'softplus':
                return nn.Softplus()
            elif act == 'elu':
                return nn.ELU()
            elif act == 'leakyrelu':
                return nn.LeakyReLU(0.2, True)
        
        # If resize is not true, apply regular transpose convolution
        # If resize is true, apply upsampling deconvolution
        if not self.resize:
            self.main = nn.Sequential(
                nn.ConvTranspose2d(in_channel, out_channel, 3, stride=1, padding=1),  # Deconvolution
                nn.GroupNorm(8, out_channel),  # Group normalization
                get_act(),  # Activation
                nn.ConvTranspose2d(out_channel, out_channel, 3, stride=1, padding=1),  # Second deconvolution
                nn.GroupNorm(8, out_channel)
            )
        else:  
            self.main = nn.Sequential(
                nn.ConvTranspose2d(in_channel, out_channel, 3, stride=1, padding=1),  # Deconvolution without resizing
                nn.GroupNorm(8, out_channel),
                get_act(),
                nn.ConvTranspose2d(out_channel, out_channel, 3, stride=2, padding=1, output_padding=1),  # Upsampling deconvolution
                nn.GroupNorm(8, out_channel)
            )
            self.residual = nn.ConvTranspose2d(in_channel, out_channel, 3, stride=2, padding=1, output_padding=1)  # Residual deconvolution for upsampling

        self.final_act = get_act()
        
    def forward(self, inputs):
        if not self.resize:  
            h = self.main(inputs)
            h += inputs
        else:  
            h = self.main(inputs)
            res = self.residual(inputs)
            h += res
        return self.final_act(h)  # Apply final activation
        
# ResNet-based score network for generative modeling
class ResScore(nn.Module):
    def __init__(self, config):  
        super(ResScore, self).__init__()
        self.nef = config.model.nef  # Number of filters in the encoder
        self.ndf = config.model.ndf  # Number of filters in the decoder
        act = 'elu'
        
        self.convs = nn.Sequential(
            nn.Conv2d(3, self.nef, 3, 1, 1), 
            ConvResBlock(self.nef, self.nef, act=act),  
            ConvResBlock(self.nef, 2 * self.nef, resize=True, act=act),  
            ConvResBlock(2 * self.nef, 2 * self.nef, act=act), 
            ConvResBlock(2 * self.nef, 4 * self.nef, resize=True, act=act),  
            ConvResBlock(4 * self.nef, 4 * self.nef, act=act),  
        )
        
        self.deconvs = nn.Sequential(
            DeconvResBlock(4 * self.ndf, 4 * self.ndf, act=act),  
            DeconvResBlock(4 * self.ndf, 2 * self.ndf, resize=True, act=act), 
            DeconvResBlock(2 * self.ndf, 2 * self.ndf, act=act),  
            DeconvResBlock(2 * self.ndf, self.ndf, resize=True, act=act), 
            DeconvResBlock(self.ndf, self.ndf, act=act),  
            nn.Conv2d(self.ndf, 3, 3, 1, 1)  
        )

    def forward(self, x):
        x = 2 * x - 1.  # Normalize input to [-1, 1]
        res = self.deconvs(self.convs(x)) 
        return res
    
class ResNetScore(nn.Module):
    def __init__(self, config):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        super().__init__()

        input_nc = output_nc = config.data.channels
        ngf = config.model.ngf * 2
        n_blocks = 6
        norm_layer = get_norm_layer('instance')
        use_dropout = False
        padding_type = 'reflect' # Padding type for convolutions
        assert (n_blocks >= 0)
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ELU()]
        
        
        n_downsampling = 1
        for i in range(n_downsampling): 
            mult = 2 ** i  
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),  
                norm_layer(ngf * mult * 2),  
                nn.ELU()  
            ]

        mult = 2 ** n_downsampling 
        for i in range(n_blocks): 
            model += [
                ResnetBlock(
                    ngf * mult, 
                    padding_type=padding_type, 
                    norm_layer=norm_layer, 
                    use_dropout=use_dropout, 
                    use_bias=use_bias
                )
            ]
        
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(
                    ngf * mult, 
                    int(ngf * mult / 2), 
                    kernel_size=3, 
                    stride=2, 
                    padding=1, 
                    output_padding=1, 
                    bias=use_bias
                ),  
                norm_layer(int(ngf * mult / 2)), 
                nn.ELU()
            ]
        
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)  
        ]
        self.model = nn.Sequential(*model)
        
    def forward(self, input):
        input = 2 * input - 1.
        return self.model(input)
    
class UNetScore(nn.Module):
    def __init__(self, config):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super().__init__()
        # construct unet structure
        input_nc = output_nc = config.data.channels
        ngf = config.model.ngf
        self.config = config
        norm_layer = get_norm_layer('instance')
        
        if config.data.image_size == 32:
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None,
                                             norm_layer=norm_layer, innermost=True)
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer)
        elif config.data.image_size == 16:
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None,
                                                 norm_layer=norm_layer, innermost=True)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf * 2, input_nc=input_nc, submodule=unet_block,
                                             outermost=True,
                                             norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        if not self.config.data.logit_transform:
            input = 2 * input - 1.
        return self.model(input)