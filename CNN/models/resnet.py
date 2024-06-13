"""
Authors : 
    Davide Galli (davide.galli@polimi.it),
    Francesco Lattari,
    Matteo Matteucci (matteo.matteucci@polimi.it),
    Davide Zoni (davide.zoni@polimi.it)
"""

import torch.nn as nn
import torch.nn.functional as F

from .custom_layers import Conv1dPadSame, MaxPool1dPadSame


class ResidualBlock(nn.Module):
    """
    ResNet basic residual block
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, downsample, use_bn, use_do, do_val, is_first_block=False):
        super().__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.downsample = downsample
        if self.downsample:
            self.stride = stride
        else:
            self.stride = 1
        self.is_first_block = is_first_block
        self.use_bn = use_bn
        self.use_do = use_do
        self.do_val = do_val

        # the first conv
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.ReLU()
        self.do1 = nn.Dropout(p=do_val)
        self.conv1 = Conv1dPadSame(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=self.stride,
            groups=self.groups)

        # the second conv
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.do2 = nn.Dropout(p=do_val)
        self.conv2 = Conv1dPadSame(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            groups=self.groups)

        self.max_pool = MaxPool1dPadSame(kernel_size=self.stride)

    def forward(self, x):
        identity = x

        # the first conv
        out = x
        if not self.is_first_block:
            if self.use_bn:
                out = self.bn1(out)
            out = self.relu1(out)
            if self.use_do:
                out = self.do1(out)
        out = self.conv1(out)

        # the second conv
        if self.use_bn:
            out = self.bn2(out)
        out = self.relu2(out)
        if self.use_do:
            out = self.do2(out)
        out = self.conv2(out)

        # if downsample, also downsample identity
        if self.downsample:
            identity = self.max_pool(identity)

        # if expand channel, also pad zeros to identity
        if self.out_channels != self.in_channels:
            identity = identity.transpose(-1, -2)
            ch1 = (self.out_channels-self.in_channels)//2
            ch2 = self.out_channels-self.in_channels-ch1
            identity = F.pad(identity, (ch1, ch2), "constant", 0)
            identity = identity.transpose(-1, -2)

        # shortcut
        out += identity

        return out


class ResNet(nn.Module):
    """
    1D ResNet encoder

    Input:
        X: (n_samples, n_channels, n_length)
        Y: (n_samples)

    Output:
        out: (n_samples)
    """

    def __init__(self, encoding_size, base_filters, kernel_size, stride, groups, n_block,
                 in_channels=1, downsample_gap=2, increasefilter_gap=4,
                 use_batch_norm=True, use_inner_do=True, inner_do_val=0.5,
                 use_final_do=True, final_do_val=0.2, gradcam=False, verbose=False):
        super(ResNet, self).__init__()
        '''
        Create new ResNet encoder module.
        encoding_size: dimension of encoder latent space
        base_filters: number of filters in the first several Conv layer, it will double at every 4 layers
        kernel_size: width of kernel
        stride: stride of kernel moving
        groups: set larget to 1 as ResNeXt
        n_block: number of blocks
        ido_val: inner blocks dropout value
        fdo_val: final layer dropout value
        '''

        self.forward_hooks = {}
        self.backward_hooks = {}

        self.verbose = verbose
        self.n_block = n_block
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.use_bn = use_batch_norm
        self.use_ido = use_inner_do
        self.ido_val = inner_do_val
        self.use_fdo = use_final_do
        self.fdo_val = final_do_val
        self.gradcam = gradcam

        self.downsample_gap = downsample_gap  # 2 for base model
        self.increasefilter_gap = increasefilter_gap  # 4 for base model

        # first block
        self.first_block_conv = Conv1dPadSame(
            in_channels=in_channels, out_channels=base_filters, kernel_size=self.kernel_size, stride=1)
        self.first_block_bn = nn.BatchNorm1d(base_filters)
        self.first_block_relu = nn.ReLU()
        out_channels = base_filters

        # residual blocks
        self.basicblock_list = nn.ModuleList()
        for i_block in range(self.n_block):
            # is_first_block
            if i_block == 0:
                is_first_block = True
            else:
                is_first_block = False
            # downsample at every self.downsample_gap blocks
            # if (i_block + 1) % self.downsample_gap == 0:
            if i_block % self.downsample_gap == 1:
                downsample = True
            else:
                downsample = False
            # in_channels and out_channels
            if is_first_block:
                in_channels = base_filters
                out_channels = in_channels
            else:
                # increase filters at every self.increasefilter_gap blocks
                in_channels = int(
                    base_filters*2**((i_block-1)//self.increasefilter_gap))
                # if (i_block + 1) % self.increasefilter_gap == 0:
                if (i_block % self.increasefilter_gap == 0) and (i_block != 0):
                    out_channels = in_channels * 2
                else:
                    out_channels = in_channels

            tmp_block = ResidualBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                groups=self.groups,
                downsample=downsample,
                use_bn=self.use_bn,
                use_do=self.use_ido,
                do_val=self.ido_val,
                is_first_block=is_first_block)
            self.basicblock_list.append(tmp_block)

        # final
        self.final_bn = nn.BatchNorm1d(out_channels)
        self.final_relu = nn.ReLU()

        if use_final_do:
            self.final_do = nn.Dropout(final_do_val)

        # store out_channels
        self.out_channels = out_channels
        self.encoding_size = encoding_size

        # hooks
        self.final_relu.register_forward_hook(self.forward_grad_cam_hook())

        self.encoding = nn.Linear(self.out_channels, encoding_size)
        self.encoding_relu = nn.ReLU()

    def make_forward_hook(self, name):
        '''
        Forward hook handlers builder.
        name: name in the forward hooks dict
        '''
        def hook(model, input, output):
            self.forward_hooks[name] = output
        return hook

    def make_backward_hook(self, name):
        '''
        Backward hook handlers builder.
        name: name in the backward hooks dict
        '''
        def hook(grad):
            self.backward_hooks[name] = grad
        return hook

    def forward_grad_cam_hook(self):
        '''
        Forward hook for Grad-CAM computation.
        '''
        return self.make_forward_hook('grad_cam')

    def backward_grad_cam_hook(self):
        '''
        Backward hook for Grad-CAM computation
        '''
        return self.make_backward_hook('grad_cam')

    def forward(self, x):
        out = x

        # first conv
        if self.verbose:
            print('input shape', out.shape)
        out = self.first_block_conv(out)
        if self.verbose:
            print('after first conv', out.shape)
        if self.use_bn:
            out = self.first_block_bn(out)
        out = self.first_block_relu(out)

        # residual blocks, every block has two conv
        for i_block in range(self.n_block):
            net = self.basicblock_list[i_block]
            if self.verbose:
                print('i_block: {0}, in_channels: {1}, out_channels: {2}, downsample: {3}'.format(
                    i_block, net.in_channels, net.out_channels, net.downsample))
            out = net(out)
            if self.verbose:
                print(out.shape)

        # final
        if self.use_bn:
            out = self.final_bn(out)
        out = self.final_relu(out)
        out = out.mean(-1)
        if self.gradcam:
            # TODO: enabled for segmentation, exception for training
            out.register_hook(self.backward_grad_cam_hook())
        if self.verbose:
            print('average global pooling', out.shape)

        # encoding
        out = self.encoding(out)
        out = self.encoding_relu(out)
        if self.use_fdo:
            out = self.final_do(out)
        if self.verbose:
            print('encoder output', out.shape)
        return out
