from torchvision import datasets, transforms
import torch.nn.functional as F
from capsule_network import Net
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch
import time


BATCH_SIZE = 512
DISP_ITER = 25
NUM_EPOCH = 10


def run_train_iter(data, target, model, optimizer):
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target)

    loss.backward()
    optimizer.step()
    return output, loss


def accuracy(y, pred):
    acc = (np.argmax(y.detach().numpy(), axis=1) == pred.detach().numpy()) / len(pred)
    # print(np.argmax(y.detach().numpy(), axis=1).shape, pred.detach().numpy().shape)
    return np.mean(acc)


def train():
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../capsnet/data",
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ])
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../capsnet/data",
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ])
        ),
        shuffle=True,
    )

    model = Net()
    optimizer = optim.Adam(params=model.parameters(), lr=1e-4)

    for epoch in range(NUM_EPOCH):
        for batch_idx, (data, target) in enumerate(train_loader):
            t1 = time.time()
            output, loss = run_train_iter(data, target, model, optimizer)
            t2 = time.time()

            if batch_idx % DISP_ITER == 0:
                acc = accuracy(output, target)

                print(
                    "\r{epoch_idx} [{batch_idx}/{num_iter}]:\tAccuracy: {acc:.3f}\tLoss: {loss:.3f}\tRuntime: {runtime:.3f}".format(
                        epoch_idx=epoch,
                        batch_idx=batch_idx,
                        num_iter=len(train_loader),
                        acc=acc,
                        loss=loss,
                        runtime=t2-t1,
                    ),
                    end="",
                    flush=True,
                )

        acc = accuracy(output, target)

        print(
            "\r{epoch_idx} [{batch_idx}/{num_iter}]:\tAccuracy: {acc}\tLoss: {loss}".format(
                epoch_idx=epoch,
                batch_idx=batch_idx,
                num_iter=len(train_loader),
                acc=acc,
                loss=loss,
            ),
            end="\n",
            flush=True
        )
    return 0


if __name__ == "__main__":
    train()
