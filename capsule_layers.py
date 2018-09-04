import torch.nn.functional as F
import torch.nn as nn
import torch


torch.manual_seed(2018)


class PrimaryCapsLayer(nn.Module):
    """Implements a primary capsule layer"""
    def __init__(self, in_channels, out_channels, num_capsules, capsule_length, num_routing_iter):
        super(PrimaryCapsLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_capsules = num_capsules
        self.capsule_length = capsule_length
        self.num_routing_iter = num_routing_iter

        self.kernel_size = 9

        self.conv_list = []

        for _ in range(self.num_capsules):
            self.conv_list += [nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=(self.kernel_size, self.kernel_size),
                stride=2,
                padding=0,
            )]

    def forward(self, x):
        x = [conv(x) for conv in self.conv_list]
        x = torch.cat(x, dim=-1)
        return self.squash(x)

    @staticmethod
    def squash(x):
        coeff = (x.norm() ** 2) / (1 + (x.norm() ** 2))
        v = coeff * (x / x.norm())
        return v


class DigitCapsLayer(nn.Module):
    """Implements a DigitCaps layer"""
    def __init__(self):
        super(DigitCapsLayer, self).__init__
        pass


class CapsuleNetwork(nn.Module):
    """Implements the capsule network described in Sabour et al (2017)"""
    def __init__(self, num_routing_iter):
        super(CapsuleNetwork, self).__init__()
        self.num_routing_iter = num_routing_iter

        self.kernel_size = 9

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=256,
            kernel_size=(self.kernel_size, self.kernel_size),
            stride=1,
            padding=0,
        )
        self.caps_1 = PrimaryCapsLayer(
            in_channels=256,
            out_channels=32,
            num_capsules=6,  # Output of primary_conv1 is [BATCH_SIZE, 256, 6, 6]
            capsule_length=8,
            num_routing_iter=3,
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.caps_1(x)
        print(x.shape)
        return


if __name__ == "__main__":
    x = torch.rand(512, 1, 28, 28)

    model = CapsuleNetwork(
        num_routing_iter=3,
    )
    model.forward(x)
