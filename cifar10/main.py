'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import sys
import models
import importlib
import logging
import re
from datetime import datetime

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--net-type', default='CifarResNetBasic', type=str,
                    help='the type of net (ResNetBasic or ResNetBottleneck or CifarResNetBasic or CifarPlainNetBasic)')
parser.add_argument('--num-blocks', default='3-3-3', type=str, help='starting net')
parser.add_argument('--batch-size', default=128, type=int, help='batch size')
parser.add_argument('--epochs', default=200, type=int, help='the number of epochs')
parser.add_argument('--print-freq', default=3910, type=int, help='the frequency to print')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

net_arch = list(map(int, args.num_blocks.split('-')))
save_path = os.path.join('./results', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
if not os.path.exists(save_path):
    os.makedirs(save_path)
else:
    raise OSError('Directory {%s} exists. Use a new one.' % save_path)
logging.basicConfig(filename=os.path.join(save_path, 'log.txt'), level=logging.INFO)
logger = logging.getLogger('main')
logger.addHandler(logging.StreamHandler())
logger.info("Saving to %s", save_path)
logger.info("Running arguments: %s", args)
best_acc = 0  # best test accuracy
trained_batchs = 0


def decay_learning_rate(optimizer):
    """Sets the learning rate to the initial LR decayed by 10"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * 0.1

# Training
def train(epoch, net, loader, optimizer, criterion, device='cuda'):
    logger.info('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    global trained_batchs
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if 0 == batch_idx % 100 or batch_idx == len(loader) - 1:
            logger.info('(%d/%d) ==> Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (
                        batch_idx + 1, len(loader), train_loss / (batch_idx + 1), 100. * correct / total, correct,
                        total))
        trained_batchs += 1
    return train_loss / len(loader), 100. * correct / total

def test(epoch, net, loader, criterion, device='cuda'):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if 0 == batch_idx % 100 or batch_idx == len(loader) - 1:
                logger.info('(%d/%d) ==> Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (batch_idx+1, len(loader), test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        logger.info('Saving the best model %.3f @ %d ...' % (acc, epoch))
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc

    return test_loss / len(loader), acc

def plot_curves(curves, save_path):
    # plotting
    clr1 = (0.5, 0., 0.)
    clr2 = (0.0, 0.5, 0.)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('Loss', color=clr1)
    ax1.tick_params(axis='y', colors=clr1)
    ax2.set_ylabel('Accuracy (%)', color=clr2)
    ax2.tick_params(axis='y', colors=clr2)

    markersize = 12
    ax1.semilogy(curves[:, 0], curves[:, 1], '--', color=clr1,
                 markersize=markersize)
    ax1.semilogy(curves[:, 0], curves[:, 3], '-', color=clr1,
                 markersize=markersize)
    ax2.plot(curves[:, 0], curves[:, 2], '--', color=clr2,
             markersize=markersize)
    ax2.plot(curves[:, 0], curves[:, 4], '-', color=clr2,
             markersize=markersize)

    # ax2.set_ylim(bottom=40, top=100)
    ax1.legend(('Train loss', 'Val loss'), loc='lower right')
    ax2.legend(('Train accuracy', 'Val accuracy', 'Val moving avg'), loc='lower left')
    fig.savefig(os.path.join(save_path, 'curves-vs-epochs.pdf'))

def main():
    device = 'cuda' # if torch.cuda.is_available() else 'cpu'
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    logger.info('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # Model
    logger.info('==> Building model..')
    net = getattr(models, args.net_type)(net_arch)
    # net = VGG('VGG19')
    # net = ResNet18()
    # net = CifarResNetBasic([1,1,1])
    # net = PreActResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = MobileNetV2()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()

    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        logger.info('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.t7')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    curves = np.zeros((args.epochs, 5))
    for epoch in range(start_epoch, start_epoch+args.epochs):
        if epoch == args.epochs / 2 or epoch == args.epochs * 3 / 4:
            logger.info('======> decaying learning rate')
            decay_learning_rate(optimizer)

        curves[epoch, 0] = epoch
        curves[epoch, 1], curves[epoch, 2] = train(epoch, net, trainloader, optimizer, criterion)
        curves[epoch, 3], curves[epoch, 4] = test(epoch, net, testloader, criterion)
        if epoch % 5 == 0:
            plot_curves(curves[:epoch+1, :], save_path)

    plot_curves(curves, save_path)
    logger.info('curves: \n {}'.format(np.array_str(curves)))
    np.savetxt(os.path.join(save_path, 'curves.dat'), curves)



if __name__ == '__main__':
    main()
    logger.info('Done!')