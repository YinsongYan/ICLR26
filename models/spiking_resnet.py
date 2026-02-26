import torch
import torch.nn as nn
from spikingjelly.activation_based import layer

__all__ = [
    'PreActResNet', 'spiking_resnet18', 'spiking_resnet34', 'spiking_resnet50', 'spiking_resnet101', 'spiking_resnet152'
]


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_channels, out_channels, stride, dropout, neuron: callable = None, **kwargs):
        super(PreActBlock, self).__init__()
        whether_bias = True
        self.bn1 = nn.BatchNorm2d(in_channels)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=whether_bias)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.dropout = layer.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, self.expansion * out_channels, kernel_size=3, stride=1, padding=1,
                               bias=whether_bias)

        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride,
                                      padding=0, bias=whether_bias)
        else:
            self.shortcut = nn.Sequential()

        self.relu1 = neuron(hids=in_channels, **kwargs)
        self.relu2 = neuron(hids=out_channels, **kwargs)

    def forward(self, x):
        x = self.relu1(self.bn1(x))
        out = self.conv1(x)
        out = self.conv2(self.dropout(self.relu2(self.bn2(out))))
        out = out + self.shortcut(x)
        return out

    # # def fire_rate(self, x):
    # def forward(self, x):
    #     # First neuron layer
    #     x, layer1_fr = self.relu1(self.bn1(x))  # Unpack spike and firing rate
    #     out = self.conv1(x)                     # Use only the spike (tensor) for the next layer
    #
    #     # Second neuron layer
    #     out, layer2_fr = self.relu2(self.bn2(out))  # Unpack spike and firing rate for the second neuron
    #     out = self.conv2(self.dropout(out))         # Use only the spike
    #     out = out + self.shortcut(x)                # Add shortcut connection
    #
    #     # Combine firing rate info for the block
    #     total_block_fr = [layer1_fr, layer2_fr]  # Each firing rate is a list of 5 values
    #     return out, total_block_fr


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_channels, out_channels, stride, dropout, neuron: callable = None, **kwargs):
        super(PreActBottleneck, self).__init__()
        whether_bias = True

        self.bn1 = nn.BatchNorm2d(in_channels)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=whether_bias)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=whether_bias)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.dropout = layer.Dropout(dropout)
        self.conv3 = nn.Conv2d(out_channels, self.expansion * out_channels, kernel_size=1, stride=1, padding=0,
                               bias=whether_bias)

        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride,
                                      padding=0, bias=whether_bias)
        else:
            self.shortcut = nn.Sequential()

        self.relu1 = neuron(hids=in_channels, **kwargs)
        self.relu2 = neuron(hids=out_channels, **kwargs)
        self.relu3 = neuron(hids=out_channels, **kwargs)

    def forward(self, x):
        x = self.relu1(self.bn1(x))

        out = self.conv1(x)
        out = self.conv2(self.relu2(self.bn2(out)))
        out = self.conv3(self.dropout(self.relu3(self.bn3(out))))

        out = out + self.shortcut(x)

        return out


class PreActResNet(nn.Module):

    def __init__(self, block, num_blocks, num_classes, dropout, neuron: callable = None, **kwargs):
        super(PreActResNet, self).__init__()
        self.num_blocks = num_blocks
        # self.data_channels = kwargs.get('c_in', 3)
        self.data_channels = 3
        self.init_channels = 64
        self.conv1 = nn.Conv2d(self.data_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], 1, dropout, neuron, **kwargs)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], 2, dropout, neuron, **kwargs)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], 2, dropout, neuron, **kwargs)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], 2, dropout, neuron, **kwargs)

        self.bn1 = nn.BatchNorm2d(512 * block.expansion)
        self.pool = nn.AvgPool2d(4)
        self.flat = nn.Flatten()
        self.drop = layer.Dropout(dropout)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

        self.relu1 = neuron(hids=512 * block.expansion, **kwargs)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, val=1)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def _make_layer(self, block, out_channels, num_blocks, stride, dropout, neuron, **kwargs):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.init_channels, out_channels, stride, dropout, neuron, **kwargs))
            self.init_channels = out_channels * block.expansion
        return nn.Sequential(*layers)


    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool(self.relu1(self.bn1(out)))
        out = self.drop(self.flat(out))
        out = self.linear(out)
        return out


    # def forward(self, x):
    #     """
    #     Handles multiple PreActBlocks in each layer, combining their outputs and firing rates.
    #     """
    #     # Initial convolution
    #     out = self.conv1(x)
    #
    #     # Process each layer and aggregate firing rates
    #     out, t_layer1_fr = self._forward_layer(out, self.layer1)  # Layer 1
    #     out, t_layer2_fr = self._forward_layer(out, self.layer2)  # Layer 2
    #     out, t_layer3_fr = self._forward_layer(out, self.layer3)  # Layer 3
    #     out, t_layer4_fr = self._forward_layer(out, self.layer4)  # Layer 4
    #
    #     # Final layers
    #     out_spike, final_fr = self.relu1(self.bn1(out))  # Only use the spike for further processing
    #     out = self.pool(out_spike)
    #     out = self.drop(self.flat(out))
    #     out = self.linear(out)
    #
    #     # # Collect firing rates from all layers
    #     # Total_fr = [t_layer1_fr, t_layer2_fr, t_layer3_fr, t_layer4_fr, final_fr]
    #
    #     # Collect firing rates from all layers
    #     Total_fr = (
    #             t_layer1_fr +  # Layer 1: 2 blocks
    #             t_layer2_fr +  # Layer 2: 2 blocks
    #             t_layer3_fr +  # Layer 3: 2 blocks
    #             t_layer4_fr +  # Layer 4: 2 blocks
    #             [final_fr]  # Final neuron layer
    #     )
    #
    #     return out, Total_fr
    #
    # def _forward_layer(self, x, layer):
    #     """
    #     Helper function to process a layer containing multiple PreActBlocks.
    #
    #     Args:
    #         x: Input tensor to the layer.
    #         layer: Sequential container with multiple PreActBlock instances.
    #
    #     Returns:
    #         Tuple of:
    #         - Output tensor after passing through all blocks in the layer.
    #         - Aggregated firing rates from all blocks in the layer as a single Tensor.
    #     """
    #     layer_frs = []  # To store firing rates from all blocks in the layer
    #     for block in layer:
    #         x, block_fr = block(x)  # Each block returns (spike, firing rate)
    #         # `block_fr` is already a list of tensors; stack it directly
    #         # block_fr_tensor = torch.stack(block_fr, dim=0)  # Stack the firing rates from the block
    #         # layer_frs.append(block_fr_tensor)
    #         layer_frs.extend(block_fr)  # Append firing rates for both neurons in the block
    #
    #     return x, layer_frs






def spiking_resnet18(neuron: callable = None, num_classes=10, neuron_dropout=0, **kwargs):
    return PreActResNet(PreActBlock, [2, 2, 2, 2], num_classes, neuron_dropout, neuron=neuron, **kwargs)


def spiking_resnet34(neuron: callable = None, num_classes=10, neuron_dropout=0, **kwargs):
    return PreActResNet(PreActBlock, [3, 4, 6, 3], num_classes, neuron_dropout, neuron=neuron, **kwargs)


def spiking_resnet50(neuron: callable = None, num_classes=10, neuron_dropout=0, **kwargs):
    return PreActResNet(PreActBottleneck, [3, 4, 6, 3], num_classes, neuron_dropout, neuron=neuron, **kwargs)


def spiking_resnet101(neuron: callable = None, num_classes=10, neuron_dropout=0, **kwargs):
    return PreActResNet(PreActBottleneck, [3, 4, 23, 3], num_classes, neuron_dropout, neuron=neuron, **kwargs)


def spiking_resnet152(neuron: callable = None, num_classes=10, neuron_dropout=0, **kwargs):
    return PreActResNet(PreActBottleneck, [3, 8, 36, 3], num_classes, neuron_dropout, neuron=neuron, **kwargs)