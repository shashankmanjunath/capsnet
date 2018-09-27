#
# Shashank Manjunath
# 9/1/2018
# Implements "Dynamic Routing Between Capsules"
# https://arxiv.org/pdf/1710.09829.pdf
#

from capsule_layer_test import CapsuleNet
from torchvision import datasets, transforms
from capsule_layers import CapsuleNetwork
from capsule_loss import CapsuleLoss
import torch.optim as optim
import numpy as np
import torch
import time


BATCH_SIZE = 32
DISP_ITER = 1
NUM_EPOCH = 1000


torch.manual_seed(2018)


def run_train_iter(data, target, model, optimizer, loss_func):
    optimizer.zero_grad()
    output_classification, output_reconstruction = model(data)

    # Need to convert given target labels to onehot encoding
    target_ = torch.unsqueeze(target, 1)
    target_onehot = torch.cuda.FloatTensor(BATCH_SIZE, 10).zero_()
    target_onehot.scatter_(1, target_, 1)

    loss = loss_func(data, target_onehot, output_classification, output_reconstruction)
    loss.backward()
    optimizer.step()
    return output_classification, loss


def run_eval_iter(test_loader, model, loss_func):
    full_acc = []
    full_loss = []

    with torch.no_grad():
        for eval_idx, (data, target) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()

            output_classification, output_reconstruction = model(data)

            # Need to convert given target labels to onehot encoding
            target_ = torch.unsqueeze(target, 1)
            target_onehot = torch.cuda.FloatTensor(BATCH_SIZE, 10).zero_()
            target_onehot.scatter_(1, target_, 1)

            full_loss += [loss_func(data, target_onehot, output_classification, output_reconstruction)]
            full_acc += [accuracy(output_classification, target)]
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
        drop_last=True,
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
        drop_last=True,
    )

    # device = torch.device("cuda:0")

    model = CapsuleNetwork(batch_size=BATCH_SIZE, num_routing_iter=3)
    model.cuda()
    optimizer = optim.Adam(params=model.parameters())

    model2 = CapsuleNet()
    model2.cuda()

    capsule_loss = CapsuleLoss()

    for epoch in range(NUM_EPOCH):
        t_start = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()

            t1 = time.time()
            output, loss = run_train_iter(data, target, model, optimizer, capsule_loss)
            # output2, loss2 = run_train_iter(data, target, model2, optimizer2, capsule_loss)
            t2 = time.time()

            if batch_idx % DISP_ITER == 0 or batch_idx == (len(train_loader) - 1):
                acc = accuracy(output, target)

                # print("\n")

                print(
                    "\rEpoch {epoch_idx}: [{batch_idx}/{num_iter}]:\tAccuracy: {acc:.5f}\tLoss: {loss:.3f}\tRuntime: "
                    "{runtime:.3f}".format(
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

        test_acc, test_loss = run_eval_iter(test_loader, model, capsule_loss)

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
