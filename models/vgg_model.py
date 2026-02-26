import torch
from spikingjelly.activation_based import layer

__all__ = [
    'vggsnn', 'snn5', 'snn5_noAP'
]

from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SNN5(nn.Module):
    def __init__(self, neuron, num_classes=10, dropout=0.0, **kwargs):
        super(SNN5, self).__init__()
        pool = nn.Sequential(nn.AvgPool2d(2))
        self.features = nn.Sequential(
            Layer(3, 16, 3, 1, 1, neuron, **kwargs),
            Layer(16, 64, 5, 1, 1, neuron, **kwargs),
            pool,
            Layer(64, 128, 5, 1, 1, neuron, **kwargs),
            pool,
            Layer(128, 256, 5, 1, 1, neuron, **kwargs),
            pool,
            Layer(256, 512, 3, 1, 1, neuron, **kwargs),
            pool,
        )
        W = int(32 / 2 / 2 / 2 / 2 / 2)

        self.classifier = nn.Linear(512 * W * W, num_classes)
        self.drop = layer.Dropout(dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        x = self.features(input)
        # print(x.shape)
        x = self.drop(torch.flatten(x, start_dim=-3, end_dim=-1))
        x = self.classifier(x)
        return x


# use for Figure.2
class SNN5_noAP(nn.Module):
    def __init__(self, neuron, num_classes=10, dropout=0.0, **kwargs):
        super(SNN5_noAP, self).__init__()
        pool = nn.Sequential(nn.AvgPool2d(2))
        # pool = APLayer(2)
        self.features = nn.Sequential(
            Layer(3, 16, 3, 1, 1, neuron, **kwargs),
            Layer(16, 64, 5, 2, 1, neuron, **kwargs),
            Layer(64, 128, 5, 2, 1, neuron, **kwargs),
            Layer(128, 256, 5, 4, 1, neuron, **kwargs),
            Layer(256, 256, 3, 2, 1, neuron, **kwargs),
        )
        # W = int(32 / 2 / 2 / 2 / 4 /  2)
        # if "fc_hw" in kwargs:
        #     W = int(kwargs["fc_hw"] / 2 / 2 / 2 / 2 / 2)

        self.classifier = nn.Linear(256, num_classes)
        self.drop = layer.Dropout(dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        x = self.features(input)
        x = self.drop(torch.flatten(x, start_dim=-3, end_dim=-1))
        x = self.classifier(x)
        return x


def snn5(neuron: callable = None, num_classes=10, neuron_dropout=0.0, **kwargs):
    return SNN5(neuron=neuron, num_classes=num_classes, dropout=neuron_dropout, **kwargs)


def snn5_noAP(neuron: callable = None, num_classes=10, neuron_dropout=0.0, **kwargs):
    return SNN5_noAP(neuron=neuron, num_classes=num_classes, dropout=neuron_dropout, **kwargs)


class SeqToANNContainer(nn.Module):
    # This code is form spikingjelly
    def __init__(self, *args):
        super().__init__()
        if len(args) == 1:
            self.module = args[0]
        else:
            self.module = nn.Sequential(*args)

    def forward(self, x_seq: torch.Tensor):
        y_shape = [x_seq.shape[0], x_seq.shape[1]]
        y_seq = self.module(x_seq.flatten(0, 1).contiguous())
        y_shape.extend(y_seq.shape[1:])
        return y_seq.view(y_shape)


class TEBN(nn.Module):
    def __init__(self, out_plane, eps=1e-5, momentum=0.1):
        super(TEBN, self).__init__()
        self.bn = SeqToANNContainer(nn.BatchNorm2d(out_plane))
        self.p = nn.Parameter(torch.ones(10, 1, 1, 1, 1, device=device))
    def forward(self, input):
        y = self.bn(input)
        # print('y size: ', y.size())
        # y = y.transpose(0, 1).contiguous()  # NTCHW  TNCHW
        # print('y size: ', y.size())
        y = y * self.p
        # y = y.contiguous().transpose(0, 1)  # TNCHW  NTCHW
        return y


class TEBNLayer(nn.Module):  # baseline+TN
    def __init__(self, in_plane, out_plane, kernel_size, stride, padding, neuron, dropout=0.0, **kwargs):
        super(TEBNLayer, self).__init__()
        self.fwd = SeqToANNContainer(
            nn.Conv2d(in_plane, out_plane, kernel_size, stride, padding),
        )
        self.bn = TEBN(out_plane)
        self.act = neuron(hids=out_plane, **kwargs)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y = self.fwd(x)
        y = self.bn(y)
        y = self.act(y)
        y = self.drop(y)
        return y



class Layer(nn.Module):
    def __init__(self, in_plane, out_plane, kernel_size, stride, padding, neuron, dropout=0.5, **kwargs):
        super(Layer, self).__init__()
        # self.norm_layer = TEBN(out_plane) if enable_TEBN else nn.BatchNorm2d(out_plane)
        # self.norm_layer = nn.BatchNorm2d(out_plane)
        # self.fwd = nn.Sequential(
        #     SeqToANNContainer(nn.Conv2d(in_plane, out_plane, kernel_size, stride, padding)),
        #     SeqToANNContainer(nn.BatchNorm2d(out_plane)),
        # )
        # self.droprate = kwargs.get('dropout', '0.0')
        self.fwd = nn.Sequential(
            nn.Conv2d(in_plane, out_plane, kernel_size, stride, padding),
            nn.BatchNorm2d(out_plane),
        )
        self.act = neuron(hids=out_plane, **kwargs)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fwd(x)
        x = self.act(x)
        x = self.drop(x)
        # print(x.shape)
        return x


class VGGSNN(nn.Module):
    def __init__(self, neuron, num_classes=10, neuron_dropout=0.0, **kwargs):
        super(VGGSNN, self).__init__()
        # pool = SeqToANNContainer(nn.AvgPool2d(2))
        # last_pool = SeqToANNContainer(nn.AdaptiveAvgPool2d(output_size=(3, 3)))
        pool = nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=2))
        last_pool = nn.AdaptiveAvgPool2d(output_size=(3, 3))
        self.avgpool = nn.Identity()
        # self.avgpool = nn.AdaptiveAvgPool2d((3, 3))
        # self.fc = nn.Linear(4608, num_classes)
        # pool = APLayer(2)
        self.features = nn.Sequential(
            Layer(2, 64, 3, 1, 1, neuron, **kwargs),
            Layer(64, 128, 3, 1, 1, neuron, **kwargs),
            pool,
            Layer(128, 256, 3, 1, 1, neuron,  **kwargs),
            Layer(256, 256, 3, 1, 1, neuron, **kwargs),
            pool,
            Layer(256, 512, 3, 1, 1, neuron, **kwargs),
            Layer(512, 512, 3, 1, 1, neuron,  **kwargs),
            pool,
            Layer(512, 512, 3, 1, 1, neuron, **kwargs),
            Layer(512, 512, 3, 1, 1, neuron, **kwargs),
            last_pool,
        )
        W = int(48 / 2 / 2 / 2 / 2)
        if "fc_hw" in kwargs:
            W = int(kwargs["fc_hw"] / 2 / 2 / 2 / 2)
        # self.T = 4
        # self.classifier = SeqToANNContainer(nn.Linear(512 * W * W, 10))
        # self.classifier = nn.Linear(512 * W * W, num_classes)
        # self.drop = layer.Dropout(neuron_dropout)
        # self.classifier = nn.Sequential(nn.Dropout(0.25), nn.Linear(512 * W * W, num_classes))
        self.classifier = nn.Sequential(nn.Dropout(0.25), layer.Linear(512 * 3 * 3, num_classes))
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        
        # 

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.Linear):
        #         nn.init.normal_(m.weight, 0, 0.01)
        #         nn.init.constant_(m.bias, 0)



    def forward(self, input):
        x = self.features(input)
        # x = torch.flatten(x, 2)
        # x = self.drop(torch.flatten(x, start_dim=-3, end_dim=-1))
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=-3, end_dim=-1)
        x = self.classifier(x)
        return x



# class VGGSNN(nn.Module):
#     def __init__(self, neuron, num_classes=10, neuron_dropout=0.0, **kwargs):
#         super(VGGSNN, self).__init__()
#         pool = SeqToANNContainer(nn.AvgPool2d(kernel_size=2, stride=2))
#         # pool = SeqToANNContainer(nn.AvgPool2d(2))
#         last_pool = SeqToANNContainer(nn.AdaptiveAvgPool2d(output_size=(3, 3)))
#         self.step_mode = kwargs.get('step_mode', 's')
#         # pool = APLayer(2)
#         # self.features = nn.Sequential(
#         #     TEBNLayer(2, 64, 3, 1, 1, neuron, **kwargs),
#         #     TEBNLayer(64, 128, 3, 1, 1, neuron, **kwargs),
#         #     pool,
#         #     TEBNLayer(128, 256, 3, 1, 1, neuron, **kwargs),
#         #     TEBNLayer(256, 256, 3, 1, 1, neuron, **kwargs),
#         #     pool,
#         #     TEBNLayer(256, 512, 3, 1, 1, neuron, **kwargs),
#         #     TEBNLayer(512, 512, 3, 1, 1, neuron, **kwargs),
#         #     pool,
#         #     TEBNLayer(512, 512, 3, 1, 1, neuron, **kwargs),
#         #     TEBNLayer(512, 512, 3, 1, 1, neuron, **kwargs),
#         #     last_pool,
#         # )
#
#         self.features = nn.Sequential(
#             Layer(2, 64, 3, 1, 1, neuron, **kwargs),
#             Layer(64, 128, 3, 1, 1, neuron, **kwargs),
#             pool,
#             Layer(128, 256, 3, 1, 1, neuron, **kwargs),
#             Layer(256, 256, 3, 1, 1, neuron, **kwargs),
#             pool,
#             Layer(256, 512, 3, 1, 1, neuron, **kwargs),
#             Layer(512, 512, 3, 1, 1, neuron, **kwargs),
#             pool,
#             Layer(512, 512, 3, 1, 1, neuron, **kwargs),
#             Layer(512, 512, 3, 1, 1, neuron, **kwargs),
#             last_pool,
#         )
#
#         W = int(48 / 2 / 2 / 2 / 2)
#         # self.T = 10
#         # self.classifier = nn.Sequential(nn.Dropout(0.25), SeqToANNContainer(nn.Linear(512 * W * W, num_classes)))
#         self.classifier = nn.Sequential(nn.Dropout(0.25), SeqToANNContainer(nn.Linear(512 * 3 * 3, num_classes)))
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#
#     def forward(self, input):
#         # input = add_dimention(input, self.T)
#         x = self.features(input)
#         # print(f'x shape: {x.shape}')  # torch.Size([10, 32, 512, 3, 3])
#         if self.step_mode == 's':
#             x = torch.flatten(x, 1)
#         elif self.step_mode == 'm':
#             x = torch.flatten(x, 2)
#         # print(f'after flatten x shape: {x.shape}')   #  torch.Size([10, 32, 4608])
#         x = self.classifier(x)
#         return x


class VGGSNNwoAP(nn.Module):
    def __init__(self, neuron, num_classes=10, neuron_dropout=0.0, **kwargs):
        super(VGGSNNwoAP, self).__init__()
        self.features = nn.Sequential(
            Layer(2, 64, 3, 1, 1, neuron, **kwargs),
            Layer(64, 128, 3, 2, 1, neuron, **kwargs),
            Layer(128, 256, 3, 1, 1, neuron, **kwargs),
            Layer(256, 256, 3, 2, 1, neuron, **kwargs),
            Layer(256, 512, 3, 1, 1, neuron, **kwargs),
            Layer(512, 512, 3, 2, 1, neuron, **kwargs),
            Layer(512, 512, 3, 1, 1, neuron, **kwargs),
            Layer(512, 512, 3, 2, 1, neuron, **kwargs),
        )
        W = int(48 / 2 / 2 / 2 / 2)
        if "fc_hw" in kwargs:
            W = int(kwargs["fc_hw"] / 2 / 2 / 2 / 2)

        # self.T = 4
        # self.classifier = SeqToANNContainer(nn.Linear(512 * W * W, 10))
        self.classifier = nn.Linear(512 * W * W, num_classes)
        self.drop = layer.Dropout(neuron_dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        # print(input.shape)
        x = self.features(input)
        # print(x.shape)
        x = self.drop(torch.flatten(x, start_dim=-3, end_dim=-1))

        x = self.classifier(x)
        return x


def vggsnn(neuron: callable = None, num_classes=10, neuron_dropout=0.0, **kwargs):
    return VGGSNN(neuron=neuron, num_classes=num_classes, dropout=neuron_dropout, **kwargs)


if __name__ == '__main__':
    # model = VGGSNNwoAP()
    from modules.neuron import ComplementaryLIFNeuron
    from thop import profile

    model = snn5_noAP(neuron=ComplementaryLIFNeuron)
    input = torch.randn(1, 3, 32, 32)
    flops, params = profile(model, inputs=(input,))
    print(model)
