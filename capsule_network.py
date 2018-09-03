#
# Shashank Manjunath
# 9/1/2018
# Implements "Dynamic Routing Between Capsules"
# https://arxiv.org/pdf/1710.09829.pdf
#

from torchvision import datasets, transforms
import torch.nn.functional as F
from baseline_network import BaselineNetwork
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch
import time


BATCH_SIZE = 8192
DISP_ITER = 1
NUM_EPOCH = 100


def run_train_iter(data, target, model, optimizer):
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target)

    loss.backward()
    optimizer.step()
    return output, loss


def run_eval_iter(test_loader, model):
    full_acc = []
    full_loss = []

    with torch.no_grad():
        for _, (data, target) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()

            output = model(data)

            full_loss += [F.nll_loss(output, target)]
            full_acc += [accuracy(output, target)]

    return np.mean(full_acc), np.mean(full_loss)


def accuracy(y, pred):
    acc = np.argmax(y.cpu().detach().numpy(), axis=1) == pred.cpu().detach().numpy()
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
        shuffle=False,
        batch_size=BATCH_SIZE,
    )

    device = torch.device("cuda:0")

    model = Net()
    model.cuda()

    optimizer = optim.Adam(params=model.parameters(), lr=1e-4)

    for epoch in range(NUM_EPOCH):
        t_start = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()

            t1 = time.time()
            output, loss = run_train_iter(data, target, model, optimizer)
            t2 = time.time()

            if batch_idx % DISP_ITER == 0 or batch_idx == (len(train_loader) - 1):
                acc = accuracy(output, target)

                print(
                    "\r{epoch_idx} [{batch_idx}/{num_iter}]:\tAccuracy: {acc:.5f}\tLoss: {loss:.3f}\tRuntime: {runtime:.3f}".format(
                        epoch_idx=epoch,
                        batch_idx=batch_idx+1,
                        num_iter=len(train_loader),
                        acc=acc,
                        loss=loss,
                        runtime=t2-t1,
                    ),
                    end="",
                    flush=True,
                )

        print("", end="\n")
        t_end = time.time()

        test_acc, test_loss = run_eval_iter(test_loader, model)

        print(
            "Test Accuracy: {test_acc:.3f}\tTest Loss: {loss:.3f}\tEpoch Runtime: {runtime:.3f}".format(
                test_acc=test_acc,
                loss=test_loss,
                runtime=t_end-t_start,
            ),
            end="\n",
            flush=True
        )
    return 0


if __name__ == "__main__":
    train()
