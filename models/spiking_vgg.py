import torch
import torch.nn as nn
from copy import deepcopy
from spikingjelly.activation_based import layer

__all__ = [
    'SpikingVGGBN', 'spiking_vgg11_bn', 'spiking_vgg13_bn', 'spiking_vgg16_bn', 'spiking_vgg19_bn', 'spiking_vgg11_bn_wdrop'
]

cfg = {

    'VGG11': [
        [64, 'M'],
        [128, 'M'],
        [256, 256, 'M'],
        [512, 512, 'M'],
        [512, 512, 'M']
    ],
    'VGG13': [
        [64, 64, 'M'],
        [128, 128, 'M'],
        [256, 256, 'M'],
        [512, 512, 'M'],
        [512, 512, 'M']
    ],
    'VGG16': [
        [64, 64, 'M'],
        [128, 128, 'M'],
        [256, 256, 256, 'M'],
        [512, 512, 512, 'M'],
        [512, 512, 512, 'M']
    ],
    'VGG19': [
        [64, 64, 'M'],
        [128, 128, 'M'],
        [256, 256, 256, 256, 'M'],
        [512, 512, 512, 512, 'M'],
        [512, 512, 512, 512, 'M']
    ]
}


class SpikingVGGBN(nn.Module):
    def __init__(self, vgg_name, neuron: callable = None, dropout=0.0, num_classes=10, **kwargs):
        super(SpikingVGGBN, self).__init__()
        self.whether_bias = True
        self.init_channels = kwargs.get('c_in', 2)

        self.layer1 = self._make_layers(cfg[vgg_name][0], dropout, neuron, **kwargs)
        self.layer2 = self._make_layers(cfg[vgg_name][1], dropout, neuron, **kwargs)
        self.layer3 = self._make_layers(cfg[vgg_name][2], dropout, neuron, **kwargs)
        self.layer4 = self._make_layers(cfg[vgg_name][3], dropout, neuron, **kwargs)
        self.layer5 = self._make_layers(cfg[vgg_name][4], dropout, neuron, **kwargs)

        self.avgpool = nn.AdaptiveAvgPool2d((3, 3))

        self.classifier = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(512 * 3 * 3, num_classes),
        )

        # self.classifier = nn.Sequential(
        #     nn.Linear(512 * 7 * 7, 4096),
        #     neuron(hids=4096, **deepcopy(kwargs)),
        #     nn.Dropout(),
        #     nn.Linear(4096, 4096),
        #     neuron(hids=4096, **deepcopy(kwargs)),
        #     nn.Dropout(),
        #     nn.Linear(4096, num_classes),
        # )


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    # def _make_layers(self, cfg, dropout, neuron, **kwargs):
    #     layers = []
    #     for x in cfg:
    #         if x == 'M':
    #             layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
    #             # layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    #         else:
    #             layers.append(nn.Conv2d(self.init_channels, x, kernel_size=3, padding=1, bias=self.whether_bias))
    #             layers.append(nn.BatchNorm2d(x))
    #             # kwargs["l_i"] += 1
    #             # layers.append(neuron(**kwargs))
    #             layers.append(neuron(hids=x, **kwargs))
    #             layers.append(layer.Dropout(dropout))
    #             self.init_channels = x
    #     return nn.Sequential(*layers)

    def _make_layers(self, cfg, dropout, neuron, batch_norm=True, **kwargs):
        layers = []
        for v in cfg:
            if v == 'M':
                # layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(self.init_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), neuron(hids=v, **deepcopy(kwargs))]
                else:
                    layers += [conv2d, neuron(hids=v, **deepcopy(kwargs))]
                self.init_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.avgpool(out)
        # print(f'avg out shape: {out.shape}')
        out = torch.flatten(out, 1)
        # print(f'flatten out shape: {out.shape}')
        out = self.classifier(out)
        # print(f'out shape: {out.shape}')
        # for i, layer in enumerate(self.classifier):
        #     out = layer(out)
        #     print(f'After layer {i} ({layer.__class__.__name__}): {out.shape}')

        return out



class SpikingVGGBNwdrop(nn.Module):
    def __init__(self, vgg_name, neuron: callable = None, dropout=0.0, num_classes=10, **kwargs):
        super(SpikingVGGBNwdrop, self).__init__()
        self.whether_bias = True
        self.init_channels = kwargs.get('c_in', 2)

        self.layer1 = self._make_layers(cfg[vgg_name][0], dropout, neuron, **kwargs)
        self.layer2 = self._make_layers(cfg[vgg_name][1], dropout, neuron, **kwargs)
        self.layer3 = self._make_layers(cfg[vgg_name][2], dropout, neuron, **kwargs)
        self.layer4 = self._make_layers(cfg[vgg_name][3], dropout, neuron, **kwargs)
        self.layer5 = self._make_layers(cfg[vgg_name][4], dropout, neuron, **kwargs)

        self.avgpool = nn.AdaptiveAvgPool2d((3, 3))

        self.classifier = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(512 * 3 * 3, num_classes),
        )

        # self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        #
        # self.classifier = nn.Sequential(
        #     # nn.Dropout(0.25),
        #     nn.Linear(512 * 7 * 7, num_classes),
        # )

        # self.classifier = nn.Sequential(
        #     nn.Linear(512 * 7 * 7, 4096),
        #     neuron(hids=4096, **deepcopy(kwargs)),
        #     nn.Dropout(),
        #     nn.Linear(4096, 4096),
        #     neuron(hids=4096, **deepcopy(kwargs)),
        #     nn.Dropout(),
        #     nn.Linear(4096, num_classes),
        # )


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layers(self, cfg, dropout, neuron, **kwargs):
        layers = []
        for x in cfg:
            if x == 'M':
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
                # layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(self.init_channels, x, kernel_size=3, padding=1, bias=self.whether_bias))
                layers.append(nn.BatchNorm2d(x))
                # kwargs["l_i"] += 1
                # layers.append(neuron(**kwargs))
                layers.append(neuron(hids=x, **kwargs))
                layers.append(nn.Dropout(dropout))
                self.init_channels = x
        return nn.Sequential(*layers)

    # def _make_layers(self, cfg, dropout, neuron, batch_norm=True, **kwargs):
    #     layers = []
    #     for v in cfg:
    #         if v == 'M':
    #             # layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
    #             layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
    #         else:
    #             conv2d = nn.Conv2d(self.init_channels, v, kernel_size=3, padding=1)
    #             if batch_norm:
    #                 layers += [conv2d, nn.BatchNorm2d(v), neuron(hids=v, **deepcopy(kwargs))]
    #             else:
    #                 layers += [conv2d, neuron(hids=v, **deepcopy(kwargs))]
    #             self.init_channels = v
    #     return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.avgpool(out)
        # print(f'avg out shape: {out.shape}')
        out = torch.flatten(out, 1)
        # print(f'flatten out shape: {out.shape}')
        out = self.classifier(out)
        # print(f'out shape: {out.shape}')
        # for i, layer in enumerate(self.classifier):
        #     out = layer(out)
        #     print(f'After layer {i} ({layer.__class__.__name__}): {out.shape}')

        return out



def spiking_vgg11_bn_wdrop(neuron: callable = None, num_classes=10, neuron_dropout=0.0, **kwargs):
    return SpikingVGGBNwdrop('VGG11', neuron=neuron, dropout=neuron_dropout, num_classes=num_classes, **kwargs)




def spiking_vgg9_bn(neuron: callable = None, num_classes=10, neuron_dropout=0.0, **kwargs):
    return SpikingVGGBN('VGG9', neuron=neuron, dropout=neuron_dropout, num_classes=num_classes, **kwargs)


def spiking_vgg11_bn(neuron: callable = None, num_classes=10, neuron_dropout=0.0, **kwargs):
    return SpikingVGGBN('VGG11', neuron=neuron, dropout=neuron_dropout, num_classes=num_classes, **kwargs)


def spiking_vgg13_bn(neuron: callable = None, num_classes=10, neuron_dropout=0.0, **kwargs):
    return SpikingVGGBN('VGG13', neuron=neuron, dropout=neuron_dropout, num_classes=num_classes, **kwargs)


def spiking_vgg16_bn(neuron: callable = None, num_classes=10, neuron_dropout=0.0, **kwargs):
    return SpikingVGGBN('VGG16', neuron=neuron, dropout=neuron_dropout, num_classes=num_classes, **kwargs)


def spiking_vgg19_bn(neuron: callable = None, num_classes=10, neuron_dropout=0.0, **kwargs):
    return SpikingVGGBN('VGG19', neuron=neuron, dropout=neuron_dropout, num_classes=num_classes, **kwargs)