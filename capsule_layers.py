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
            self.conv_list += [nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=(self.kernel_size, self.kernel_size),
                stride=2,
                padding=0,
            )]

    def forward(self, x):
        x = [conv(x) for conv in self.conv_list]
        x = [torch.unsqueeze(t, -1) for t in x]
        x = torch.cat(x, dim=-1)
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
            32,
            6,
            8,
            8,
        ))

        self.digit_weights = []
        for idx in range(self.num_classes):
            self.digit_weights += [nn.Parameter(torch.randn(8, 16))]

    def forward(self, x):
        priors = x @ self.weight  # Matrix multiplication

        # Reshaping so that shape is (BATCH_SIZE, num_capsules, capsule_length)
        priors_inter_size = 1

        for idx in range(1, len(priors.shape)-1):
            priors_inter_size = priors_inter_size * priors.shape[idx]
        priors = priors.reshape(priors.shape[0], priors_inter_size, priors.shape[-1])

        b = torch.autograd.Variable(torch.zeros(priors.size()))

        for iter in range(self.num_routing_iter):
            # Step 1
            c = self._softmax(b)

            # Step 2
            s_j = c * priors
            s_j = s_j.sum(dim=1, keepdim=True)

            # Step 3
            v_j = self._squash(s_j)

            # Step 4
            b = b + (priors * v_j).sum(dim=-1, keepdim=True)

        # Reshaping v_j to get rid of dimension 1
        v_j = v_j.reshape(v_j.shape[0], v_j.shape[2])

        # Getting interior size of v_j
        digits = [F.relu(v_j @ weight) for weight in self.digit_weights]
        digits = torch.cat(digits, dim=0)
        return digits

    @staticmethod
    def _squash(x):
        coeff = (x.norm() ** 2) / (1 + (x.norm() ** 2))
        v = coeff * (x / x.norm())
        return v

    @staticmethod
    def _softmax(x):
        # Done this way for numerical stability
        exp_val = torch.exp(x)
        return exp_val / torch.sum(exp_val)


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
            num_routing_iter=3,
        )

        self.digit_caps = DigitCapsLayer(
            in_channels=8,
            out_channels=16,
            num_capsules=32,
            capsule_length=32*6*6,
            num_classes=10,
            num_routing_iter=3,
        )
        self.linear_1 = nn.Linear(16*10, 512)
        self.linear_2 = nn.Linear(512, 1024)
        self.linear_3 = nn.Linear(1024, 784)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.caps_1(x)
        x = self.digit_caps(x)

        x = x.reshape(self.batch_size, 160)

        x = F.relu(self.linear_1(x))
        x = F.relu(self.linear_2(x))
        x = torch.sigmoid(self.linear_3(x))
        x = x.reshape(28, 28)
        return x


if __name__ == "__main__":
    x = torch.rand(1, 1, 28, 28)

    model = CapsuleNetwork(
        batch_size=1,
        num_routing_iter=3,
    )

    model.forward(x)
    print(x.shape)
