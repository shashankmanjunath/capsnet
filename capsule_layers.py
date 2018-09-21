import torch.nn.functional as F
import torch.nn as nn
import torch


torch.manual_seed(2018)


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PrimaryCapsLayer(nn.Module):
    """ Implements a primary capsule layer """
    def __init__(self, in_channels, out_channels, num_capsules, capsule_length, num_routing_iter):
        super(PrimaryCapsLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_capsules = num_capsules
        self.capsule_length = capsule_length
        self.num_routing_iter = num_routing_iter

        self.kernel_size = 9

        self.conv_list = []

        for _ in range(self.capsule_length):
            self.conv_list += [
                nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=(self.kernel_size, self.kernel_size),
                    stride=2,
                    padding=0,
                ).cuda()
            ]
        self.conv_list = nn.ModuleList(self.conv_list)

    def forward(self, x):
        x = [conv(x) for conv in self.conv_list]
        x = torch.cat([t.reshape(t.shape[0], -1, 1) for t in x], dim=-1)  # Putting list of tensors into correct shape
        return self._squash(x)

    @staticmethod
    def _squash(x):
        coeff = (x.norm() ** 2) / (1 + (x.norm() ** 2))
        v = coeff * (x / x.norm())
        return v


class DigitCapsLayer(nn.Module):
    """ Implements a DigitCaps layer """
    def __init__(self, in_channels, out_channels, num_capsules, capsule_length, num_classes, num_routing_iter):
        super(DigitCapsLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_capsules = num_capsules
        self.capsule_length = capsule_length
        self.num_classes = num_classes
        self.num_routing_iter = num_routing_iter

        self.weight = nn.Parameter(torch.randn(
            self.num_capsules,      # 10
            self.in_channels,       # 1152
            self.capsule_length,    # 8
            self.out_channels,      # 16
        ))

        self.digit_weights = []
        for idx in range(self.num_classes):
            self.digit_weights += [nn.Parameter(torch.randn(self.capsule_length, 16)).cuda()]

    def forward(self, x):
        priors = x[None, :, :, None, :] @ self.weight[:, None, :, :, :]  # Matrix multiplication

        b = torch.autograd.Variable(torch.zeros(priors.size()), requires_grad=True).cuda()

        for iter in range(self.num_routing_iter):
            # Step 1
            c = F.softmax(b)

            # Step 2
            s_j = c * priors
            s_j = s_j.sum(dim=2, keepdim=True)

            # Step 3
            v_j = self._squash(s_j)

            # Step 4
            b = b + (priors * v_j).sum(dim=-1, keepdim=True)
        return v_j

    @staticmethod
    def _squash(x):
        coeff = (x.norm() ** 2) / (1 + (x.norm() ** 2))
        v = coeff * (x / x.norm())
        return v


class CapsuleNetwork(nn.Module):
    """ Implements the capsule network described in Sabour et al (2017) """
    def __init__(self, batch_size, num_routing_iter):
        super(CapsuleNetwork, self).__init__()
        self.batch_size = batch_size
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
            num_capsules=6,
            capsule_length=8,
            num_routing_iter=self.num_routing_iter,
        )

        self.digit_caps = DigitCapsLayer(
            in_channels=32 * 6 * 6,
            out_channels=16,
            num_capsules=10,
            capsule_length=8,
            num_classes=10,
            num_routing_iter=self.num_routing_iter,
        )
        self.linear_1 = nn.Linear(16*10, 512)
        self.linear_2 = nn.Linear(512, 1024)
        self.linear_3 = nn.Linear(1024, 784)

    def forward(self, x):
        x = F.relu(self.conv1(x), inplace=True)
        x = self.caps_1(x)
        # TODO: Debug from here down
        x = self.digit_caps(x)

        # Layers if we want to do classification
        x_class = x.squeeze().transpose(0, 1)
        x_class = torch.abs(x_class).sum(dim=-1)
        x_class = F.softmax(x_class, dim=-1)

        # Layers if we want to do reconstruction
        # TODO: Debug from here
        x_recon = x.reshape(self.batch_size, 16*10)
        x_recon = F.relu(self.linear_1(x_recon))
        x_recon = F.relu(self.linear_2(x_recon))
        x_recon = torch.sigmoid(self.linear_3(x_recon))
        x_recon = x_recon.reshape(self.batch_size, 28 * 28)
        return x_class, x_recon
